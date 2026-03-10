"""
Dataset for Project 3 — Part 3 (multi-class, multi-object detection, fixed capacity).

- Loads Roboflow COCO annotations from train/valid/test.
- Keeps only classes: helmet, person, vest (COCO ids 1, 4, 5) → mapped to 0, 1, 2.
- At most one object per class: for each class, keeps the largest instance by area (w×h).
  Slot 0 = helmet, slot 1 = person, slot 2 = vest; missing class → background.
- Returns fixed-size outputs for 3 slots:
  - image: (3, H, W) float in [0, 1]
  - boxes: (3, 4) normalized (cx, cy, w, h) in [0, 1]; unused slots are zero-filled.
  - class_ids: (3,) int in {0, 1, 2, 3}; 3 = background (no object in that slot).

COCO bbox format is [x, y, w, h] pixels, top-left origin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

SplitName = Literal["train", "valid", "test"]

# COCO category IDs in the source dataset (Roboflow construction-safety)
COCO_ID_HELMET = 1
COCO_ID_PERSON = 4
COCO_ID_VEST = 5
KEEP_CATEGORY_IDS = (COCO_ID_HELMET, COCO_ID_PERSON, COCO_ID_VEST)

# Internal class IDs: 0=helmet, 1=person, 2=vest, 3=background (empty slot)
NUM_FOREGROUND_CLASSES = 3
BACKGROUND_CLASS_ID = 3
COCO_TO_INTERNAL: dict[int, int] = {
    COCO_ID_HELMET: 0,
    COCO_ID_PERSON: 1,
    COCO_ID_VEST: 2,
}
CLASS_NAMES = ("helmet", "person", "vest")
MAX_OBJECTS_PER_IMAGE = 3

_COCO_JSON_CANDIDATES = (
    "_annotations.coco.json",
    "annotations.coco.json",
    "_annotations.json",
)


def find_coco_json_in_split_dir(split_dir: Path) -> Path:
    for name in _COCO_JSON_CANDIDATES:
        p = split_dir / name
        if p.exists():
            return p
    coco_matches = sorted(split_dir.glob("*.coco.json"))
    if coco_matches:
        return coco_matches[0]
    ann_matches = sorted(split_dir.glob("*annotations*.json"))
    if ann_matches:
        return ann_matches[0]
    raise FileNotFoundError(
        f"Could not find COCO annotations under {split_dir}. "
        f"Tried: {', '.join(_COCO_JSON_CANDIDATES)}"
    )


def _split_has_coco_json(split_dir: Path) -> bool:
    try:
        find_coco_json_in_split_dir(split_dir)
        return True
    except FileNotFoundError:
        return False


@dataclass(frozen=True)
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class CocoAnn:
    id: int
    image_id: int
    category_id: int
    bbox_xywh: tuple[float, float, float, float]


@dataclass(frozen=True)
class CocoIndex:
    images: dict[int, CocoImage]
    anns_by_image: dict[int, list[CocoAnn]]
    categories: dict[int, str]


def load_coco_index(coco_json_path: Path) -> CocoIndex:
    with coco_json_path.open("r", encoding="utf-8") as f:
        coco: dict[str, Any] = json.load(f)

    images: dict[int, CocoImage] = {}
    for im in coco["images"]:
        image_id = int(im["id"])
        images[image_id] = CocoImage(
            id=image_id,
            file_name=str(im["file_name"]),
            width=int(im["width"]),
            height=int(im["height"]),
        )

    categories: dict[int, str] = {
        int(c["id"]): str(c["name"]) for c in coco.get("categories", [])
    }

    anns_by_image: dict[int, list[CocoAnn]] = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        ann = CocoAnn(
            id=int(a["id"]),
            image_id=int(a["image_id"]),
            category_id=int(a["category_id"]),
            bbox_xywh=(float(x), float(y), float(w), float(h)),
        )
        anns_by_image.setdefault(ann.image_id, []).append(ann)

    return CocoIndex(images=images, anns_by_image=anns_by_image, categories=categories)


def coco_xywh_to_normalized_cxcywh(
    bbox_xywh: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x, y, w, h = bbox_xywh
    cx = (x + w / 2.0) / float(image_width)
    cy = (y + h / 2.0) / float(image_height)
    nw = w / float(image_width)
    nh = h / float(image_height)
    return (cx, cy, nw, nh)


def normalized_cxcywh_to_xyxy_pixels(
    bbox_cxcywh: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    cx, cy, w, h = bbox_cxcywh
    cx_p = cx * float(image_width)
    cy_p = cy * float(image_height)
    w_p = w * float(image_width)
    h_p = h * float(image_height)
    x1 = cx_p - w_p / 2.0
    y1 = cy_p - h_p / 2.0
    x2 = cx_p + w_p / 2.0
    y2 = cy_p + h_p / 2.0
    return (x1, y1, x2, y2)


def auto_find_multi_obj_data_root() -> Path:
    """Locate multi_object_detection/data (folder containing train/valid/test with COCO JSON)."""
    import os

    env = os.environ.get("MULTI_OBJ_DATA_ROOT") or os.environ.get("MULTI_OBJ_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if _split_has_coco_json(p / "train"):
            return p
        raise FileNotFoundError(f"Env var points to missing dataset: {p}")

    fixed = Path(
        r"C:\Users\nirdu\Documents\Project_DL\classification-to-detection\multi_object_detection\data"
    ).resolve()
    if _split_has_coco_json(fixed / "train"):
        return fixed

    module_dir = Path(__file__).resolve().parent
    for base in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        for candidate in (
            base / "multi_object_detection" / "data",
            base / "classification-to-detection" / "multi_object_detection" / "data",
            module_dir / "data",
            module_dir.parent / "multi_object_detection" / "data",
        ):
            if candidate.exists() and _split_has_coco_json(candidate / "train"):
                return candidate

    raise FileNotFoundError(
        "Could not locate multi-object detection data directory. "
        "Pass data_root to MultiObjectDataset(...) or set MULTI_OBJ_DATA_ROOT."
    )


class MultiObjectDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Multi-class, multi-object dataset with fixed 3 slots per image.

    - Filters to helmet, person, vest only; maps to class IDs 0, 1, 2.
    - At most one instance per class: keeps the largest helmet, largest person, largest vest.
    - Slots are fixed: 0=helmet, 1=person, 2=vest. Unused slot → class_id=BACKGROUND_CLASS_ID (3), zero boxes.

    Returns
    -------
    image : (3, H, W) float, [0, 1]
    boxes : (3, 4) float, normalized (cx, cy, w, h); unused slots are zeros.
    class_ids : (3,) long, in {0, 1, 2, 3}; 3 = background.
    """

    def __init__(
        self,
        *,
        split: SplitName,
        data_root: Path | None = None,
        transform: Any | None = None,
        allow_empty: bool = True,
    ) -> None:
        self.split = split
        self.data_root = (
            data_root
            if data_root is not None
            else auto_find_multi_obj_data_root()
        )
        self.split_dir = self.data_root / split
        self.coco_path = find_coco_json_in_split_dir(self.split_dir)
        self.index = load_coco_index(self.coco_path)
        self.allow_empty = allow_empty

        # Build list of image_ids that have at least one kept object (or allow empty)
        self.image_ids = []
        for image_id in sorted(self.index.images.keys()):
            kept = self._filter_and_truncate_anns(image_id)
            if allow_empty or any(k is not None for k in kept):
                self.image_ids.append(image_id)

        self.transform = transform or T.Compose(
            [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
        )

    def _filter_and_truncate_anns(
        self, image_id: int
    ) -> list[tuple[CocoAnn, int] | None]:
        """
        At most one object per class: for each class (helmet, person, vest) keep
        the largest instance by area. Return a list of length 3 in fixed order:
        slot 0 = helmet, slot 1 = person, slot 2 = vest; missing class → None.
        """
        anns = self.index.anns_by_image.get(image_id, [])
        # best[class_id] = (ann, internal_id) for the largest of that class, or None
        best: list[tuple[CocoAnn, int] | None] = [None, None, None]

        for ann in anns:
            if ann.category_id not in COCO_TO_INTERNAL:
                continue
            internal_id = COCO_TO_INTERNAL[ann.category_id]
            x, y, w, h = ann.bbox_xywh
            if w <= 0 or h <= 0:
                continue
            area = w * h
            if best[internal_id] is None or area > (
                best[internal_id][0].bbox_xywh[2] * best[internal_id][0].bbox_xywh[3]
            ):
                best[internal_id] = (ann, internal_id)

        return best

    def _sample_random_anns(
        self, image_id: int
    ) -> list[tuple[CocoAnn, int] | None]:
        """
        Like _filter_and_truncate_anns, but for training we randomly pick
        one instance per class (if any exist) instead of always taking the largest.
        This allows the same image to contribute different objects across epochs.
        """
        anns = self.index.anns_by_image.get(image_id, [])
        # candidates[class_id] = list[(ann, internal_id)] for that class
        candidates: list[list[tuple[CocoAnn, int]]] = [[], [], []]

        for ann in anns:
            if ann.category_id not in COCO_TO_INTERNAL:
                continue
            internal_id = COCO_TO_INTERNAL[ann.category_id]
            x, y, w, h = ann.bbox_xywh
            if w <= 0 or h <= 0:
                continue
            candidates[internal_id].append((ann, internal_id))

        chosen: list[tuple[CocoAnn, int] | None] = [None, None, None]
        for class_id in range(3):
            if candidates[class_id]:
                chosen[class_id] = random.choice(candidates[class_id])

        return chosen

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[idx]
        im = self.index.images[image_id]
        img_path = self.split_dir / im.file_name
        img = Image.open(img_path).convert("RGB")

        # For training, sample a random instance per class; for eval, keep the largest.
        if self.split == "train":
            kept = self._sample_random_anns(image_id)
        else:
            kept = self._filter_and_truncate_anns(image_id)  # list of 3: (ann, id) or None

        boxes = torch.zeros(MAX_OBJECTS_PER_IMAGE, 4, dtype=torch.float32)
        class_ids = torch.full(
            (MAX_OBJECTS_PER_IMAGE,), BACKGROUND_CLASS_ID, dtype=torch.long
        )

        for slot in range(MAX_OBJECTS_PER_IMAGE):
            if kept[slot] is not None:
                ann, internal_cid = kept[slot]
                cxcywh = coco_xywh_to_normalized_cxcywh(
                    ann.bbox_xywh,
                    image_width=im.width,
                    image_height=im.height,
                )
                boxes[slot] = torch.tensor(cxcywh, dtype=torch.float32)
                class_ids[slot] = internal_cid

        image_tensor = self.transform(img)

        # Random horizontal flip for training split (update image and normalized boxes).
        if self.split == "train":
            if torch.rand(()) < 0.5:
                # Flip image tensor horizontally: (C, H, W) → flip over W.
                image_tensor = torch.flip(image_tensor, dims=[2])
                # Boxes are normalized (cx, cy, w, h) in [0,1]; horizontal flip = cx -> 1 - cx.
                if boxes.numel() > 0:
                    boxes[:, 0] = 1.0 - boxes[:, 0]

        return image_tensor, boxes, class_ids
