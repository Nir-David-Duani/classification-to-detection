"""
Dataset utilities for Project 3 — Part 2 (single-class, single-object detection).

This module is intentionally small and explicit:
- It loads Roboflow-exported COCO annotations (one bbox per image).
- It returns:
    image: torch.FloatTensor with shape (3, H, W), normalized to [0, 1]
    bbox:  torch.FloatTensor with shape (4,), in normalized (cx, cy, w, h) format in [0, 1]

COCO input bbox format is [x, y, w, h] in pixel coordinates with origin at top-left.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

SplitName = Literal["train", "valid", "test"]

_COCO_JSON_CANDIDATES = (
    "_annotations.coco.json",
    "annotations.coco.json",
    "_annotations.json",
)


def find_coco_json_in_split_dir(split_dir: Path) -> Path:
    """
    Find a COCO annotations json file inside a split directory.

    Roboflow often uses `_annotations.coco.json`, but naming can vary.
    """

    for name in _COCO_JSON_CANDIDATES:
        p = split_dir / name
        if p.exists():
            return p

    # Flexible fallbacks (prefer COCO if present)
    coco_matches = sorted(split_dir.glob("*.coco.json"))
    if coco_matches:
        return coco_matches[0]

    ann_matches = sorted(split_dir.glob("*annotations*.json"))
    if ann_matches:
        return ann_matches[0]

    raise FileNotFoundError(
        f"Could not find a COCO annotations json under split dir: {split_dir}. "
        f"Tried: {', '.join(_COCO_JSON_CANDIDATES)}"
    )


def _split_has_coco_json(split_dir: Path) -> bool:
    try:
        _ = find_coco_json_in_split_dir(split_dir)
        return True
    except FileNotFoundError:
        return False


@dataclass(frozen=True)
class CocoImage:
    """A minimal COCO image record."""

    id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class CocoAnn:
    """A minimal COCO annotation record for bbox regression."""

    id: int
    image_id: int
    category_id: int
    bbox_xywh: tuple[float, float, float, float]  # (x, y, w, h) in pixels


@dataclass(frozen=True)
class CocoIndex:
    """
    An indexed view of a COCO annotation file.

    Attributes
    ----------
    images:
        Mapping: image_id -> CocoImage
    anns_by_image:
        Mapping: image_id -> list[CocoAnn]
    categories:
        Mapping: category_id -> category_name
    """

    images: dict[int, CocoImage]
    anns_by_image: dict[int, list[CocoAnn]]
    categories: dict[int, str]


def load_coco_index(coco_json_path: Path) -> CocoIndex:
    """
    Load a COCO JSON file and build an index for fast lookup.

    Parameters
    ----------
    coco_json_path:
        Path to a Roboflow-exported COCO annotations file, typically `_annotations.coco.json`.
    """

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

    categories: dict[int, str] = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}

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
    """
    Convert COCO pixel bbox [x, y, w, h] to normalized (cx, cy, w, h) in [0, 1].
    """

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
    """
    Convert normalized (cx, cy, w, h) in [0, 1] to pixel (x1, y1, x2, y2).

    This helper is mainly for visualization and IoU calculations.
    """

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


def auto_find_safety_vest_root(start: Path | None = None) -> Path:
    """
    Try to locate the Safety Vest dataset directory.

    The returned directory is expected to contain:
        train/_annotations.coco.json
        valid/_annotations.coco.json
        test/_annotations.coco.json
    """

    env = __import__("os").environ.get("SAFETY_VEST_DATA_ROOT") or __import__("os").environ.get("SAFETY_VEST_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if _split_has_coco_json(p / "train"):
            return p
        raise FileNotFoundError(f"Env var points to missing dataset: {p}")

    # Fixed local layout (as used in this project workspace on Windows)
    fixed = Path(r"C:\Users\nirdu\Documents\Project_DL\classification-to-detection\single_object_detection\data").resolve()
    if _split_has_coco_json(fixed / "train"):
        return fixed

    start = (start or Path.cwd()).resolve()
    dataset_names = ("saftey-vest", "safety-vest")

    module_dir = Path(__file__).resolve().parent

    for base in (start, *start.parents):
        # Common layout in this project: dataset directly under `single_object_detection/data`
        direct_candidates = [
            base / "data",
            base / "single_object_detection" / "data",
            base / "classification-to-detection" / "single_object_detection" / "data",
            base / "deep-learning-classification-to-detection" / "single_object_detection" / "data",
            module_dir / "data",
            module_dir.parent / "data",
        ]
        for p in direct_candidates:
            if _split_has_coco_json(p / "train"):
                return p

        for name in dataset_names:
            candidates = [
                base / "data" / name,
                base / name,
                base / "single_object_detection" / "data" / name,
                base / "classification-to-detection" / "single_object_detection" / "data" / name,
                base / "deep-learning-classification-to-detection" / "single_object_detection" / "data" / name,
                module_dir / "data" / name,
                module_dir.parent / "data" / name,
            ]

            for p in candidates:
                if _split_has_coco_json(p / "train"):
                    return p

    raise FileNotFoundError(
        "Could not locate the Safety Vest dataset directory. "
        "Pass data_root explicitly to SafetyVestDataset(...), or set SAFETY_VEST_DATA_ROOT."
    )


class SafetyVestDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Dataset for single-class, single-object per image training (bbox regression only).

    Parameters
    ----------
    split:
        One of: 'train', 'valid', 'test'.
    data_root:
        Path to the dataset directory that contains the split folders.
        If None, the dataset will try to auto-locate the Roboflow export.
    transform:
        Optional image transform. By default, the dataset returns images as float tensors in [0, 1]
        with shape (3, H, W).

        Important: bbox coordinates are computed from the annotation JSON and are NOT updated by
        this module if you apply geometric transforms (crop/resize/flip). For Part 2, start with
        non-geometric transforms (e.g., color jitter) or handle bbox-aware transforms separately.
    sample_transform:
        Optional transform that receives **both** (PIL image, bbox_tensor) and must return
        (image_tensor, bbox_tensor). This enables bbox-aware augmentation (e.g., horizontal flip)
        without breaking labels.
    expected_num_objects:
        If set (default 1), enforce that each image has exactly this many annotations.
    """

    def __init__(
        self,
        *,
        split: SplitName,
        data_root: Path | None = None,
        transform: Any | None = None,
        sample_transform: Any | None = None,
        expected_num_objects: int = 1,
    ) -> None:
        self.split: SplitName = split
        if data_root is None:
            # Prefer the in-repo dataset location if present.
            in_repo = Path(__file__).resolve().parent / "data"
            self.data_root = in_repo if _split_has_coco_json(in_repo / "train") else auto_find_safety_vest_root()
        else:
            self.data_root = data_root
        self.split_dir = self.data_root / split
        self.coco_path = find_coco_json_in_split_dir(self.split_dir)

        self.index = load_coco_index(self.coco_path)
        self.image_ids = sorted(self.index.images.keys())

        self.expected_num_objects = expected_num_objects

        # Default: convert PIL -> float tensor in [0, 1]
        self.transform = transform if transform is not None else T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
        self.sample_transform = sample_transform

        if expected_num_objects is not None:
            # Early validation: verify that each image has exactly expected_num_objects annotations.
            for image_id in self.image_ids:
                n = len(self.index.anns_by_image.get(image_id, []))
                if n != expected_num_objects:
                    raise ValueError(
                        f"Split '{split}': image_id={image_id} has {n} annotations, expected {expected_num_objects}. "
                        "This dataset assumes single-object images for Part 2."
                    )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[idx]
        im = self.index.images[image_id]

        img_path = self.split_dir / im.file_name
        img = Image.open(img_path).convert("RGB")

        anns = self.index.anns_by_image.get(image_id, [])
        if not anns:
            raise RuntimeError(f"Image '{im.file_name}' has no annotations.")
        if self.expected_num_objects is not None and len(anns) != self.expected_num_objects:
            raise RuntimeError(
                f"Image '{im.file_name}' has {len(anns)} annotations, expected {self.expected_num_objects}."
            )

        # For Part 2 we expect exactly one object.
        ann = anns[0]
        bbox_cxcywh = coco_xywh_to_normalized_cxcywh(
            ann.bbox_xywh,
            image_width=im.width,
            image_height=im.height,
        )

        bbox_tensor = torch.tensor(bbox_cxcywh, dtype=torch.float32)
        if self.sample_transform is not None:
            image_tensor, bbox_tensor = self.sample_transform(img, bbox_tensor)
        else:
            image_tensor = self.transform(img)

        return image_tensor, bbox_tensor

