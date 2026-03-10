"""
Run multi-object detection inference on videos.

- load_model_from_checkpoint(ckpt_path, device) → model
- run_inference_for_model(ckpt_path, arch_name, videos_dir, out_dir, ...) → processes all .mp4 in videos_dir
- From CLI: run inference for a list of (ckpt_dir, arch_name) so all models write to videos_out/<arch>/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from dataset import (
    BACKGROUND_CLASS_ID,
    CLASS_NAMES,
    normalized_cxcywh_to_xyxy_pixels,
)
from model import DetectorConfig, MultiObjectDetector, NUM_SLOTS
from train import make_eval_transform

try:
    import cv2
except ImportError as e:
    raise RuntimeError("opencv-python is required for video inference. pip install opencv-python") from e


MODEL_INPUT_SIZE = 224
RESIZE_MAX_SIDE = 640
SCORE_THRESHOLD = 0.3
SLOT_COLORS = {
    0: (0, 255, 255),   # helmet - yellow
    1: (0, 255, 0),     # person - green
    2: (0, 165, 255),   # vest - orange
}


def load_model_from_checkpoint(
    ckpt_path: Path | str,
    device: str,
    *,
    fallback_arch: str = "conv_shared",
) -> tuple[MultiObjectDetector, dict[str, Any]]:
    """Load model from checkpoint. Returns (model, ckpt_info)."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("cfg") or {}
    model_cfg = cfg_dict.get("model_cfg")
    if model_cfg is None:
        model_cfg = DetectorConfig(arch=fallback_arch, output_activation="sigmoid")
    elif isinstance(model_cfg, dict):
        allowed = {"weights", "arch", "output_activation", "mlp_dims", "mlp_dropouts", "conv_drop2d"}
        model_cfg = DetectorConfig(**{k: v for k, v in model_cfg.items() if k in allowed})
    model = MultiObjectDetector(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt


def annotate_one_video(
    video_in: Path,
    video_out: Path,
    model: MultiObjectDetector,
    img_tf: Any,
    device: str,
    *,
    resize_max_side: int | None = RESIZE_MAX_SIDE,
    model_input_size: int = MODEL_INPUT_SIZE,
    score_threshold: float = SCORE_THRESHOLD,
    slot_colors: dict[int, tuple[int, int, int]] | None = None,
) -> int:
    """Process one video; draw boxes and save. Returns number of frames processed."""
    slot_colors = slot_colors or SLOT_COLORS
    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w, out_h = w0, h0
    if resize_max_side is not None and max(w0, h0) > resize_max_side:
        scale = float(resize_max_side) / float(max(w0, h0))
        out_w = int(round(w0 * scale))
        out_h = int(round(h0 * scale))
    else:
        scale = 1.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out), fourcc, float(fps), (out_w, out_h))

    frame_idx = 0
    with torch.inference_mode():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if scale != 1.0:
                frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

            H, W = frame_bgr.shape[0], frame_bgr.shape[1]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inp_rgb = cv2.resize(frame_rgb, (model_input_size, model_input_size), interpolation=cv2.INTER_AREA)
            x = img_tf(Image.fromarray(inp_rgb)).unsqueeze(0).to(device)

            pred_boxes, pred_logits = model(x)
            pred_boxes = pred_boxes.squeeze(0).detach().cpu()
            pred_logits = pred_logits.squeeze(0).detach().cpu()
            probs = torch.softmax(pred_logits, dim=-1)

            for slot in range(NUM_SLOTS):
                score, cls_id = torch.max(probs[slot], dim=-1)
                cls_id = int(cls_id)
                score = float(score)
                if cls_id == BACKGROUND_CLASS_ID or score < score_threshold:
                    continue
                cx, cy, w_box, h_box = pred_boxes[slot].tolist()
                x1, y1, x2, y2 = normalized_cxcywh_to_xyxy_pixels(
                    (cx, cy, w_box, h_box),
                    image_width=model_input_size,
                    image_height=model_input_size,
                )
                sx = float(W) / float(model_input_size)
                sy = float(H) / float(model_input_size)
                x1, x2 = x1 * sx, x2 * sx
                y1, y2 = y1 * sy, y2 * sy
                x1 = int(max(0, min(W - 1, round(x1))))
                y1 = int(max(0, min(H - 1, round(y1))))
                x2 = int(max(0, min(W - 1, round(x2))))
                y2 = int(max(0, min(H - 1, round(y2))))
                color = slot_colors.get(slot, (0, 255, 0))
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 1)
                label = f"{CLASS_NAMES[slot]}: {score:.2f}"
                ty = max(15, y1 - 8)
                cv2.putText(frame_bgr, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            writer.write(frame_bgr)
            frame_idx += 1

    cap.release()
    writer.release()
    return frame_idx


def run_inference_for_model(
    ckpt_path: Path | str,
    arch_name: str,
    videos_dir: Path | str,
    out_dir: Path | str,
    *,
    device: str | None = None,
    fallback_arch: str | None = None,
    verbose: bool = True,
) -> list[Path]:
    """
    Load checkpoint, process all .mp4 in videos_dir, write to out_dir.
    Output files: out_dir / f"{stem}_{arch_name}.mp4"
    Returns list of output paths.
    """
    videos_dir = Path(videos_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fallback_arch = fallback_arch or arch_name

    model, _ = load_model_from_checkpoint(ckpt_path, device, fallback_arch=fallback_arch)
    img_tf = make_eval_transform()

    video_files = sorted(videos_dir.glob("*.mp4"))
    if not video_files and verbose:
        print(f"No .mp4 files in {videos_dir}")
    outputs = []
    for vin in video_files:
        vout = out_dir / f"{vin.stem}_{arch_name}.mp4"
        if verbose:
            print("Annotating", vin.name, "->", vout.name)
        n = annotate_one_video(vin, vout, model, img_tf, device)
        if verbose and n % 200 == 0 and n > 0:
            print(f"  {vin.name} processed frames: {n}")
        outputs.append(vout)
    if verbose:
        print("Done. Outputs in:", out_dir.resolve())
    return outputs


# Default list: (ckpt_dir, arch_name) for "run inference for all models"
DEFAULT_MODELS = [
    ("checkpoints/part3_mlp_shared", "mlp_shared"),
    ("checkpoints/part3_conv_shared", "conv_shared"),
    ("checkpoints/part3_conv_per_slot", "conv_per_slot"),
    ("checkpoints/part3_conv_per_slot_l3", "conv_per_slot_l3"),
    ("checkpoints/part3_conv_shared_deep", "conv_shared_deep"),
    ("checkpoints/part3_conv_fpn", "conv_fpn"),
    ("checkpoints/part3_grid_shared", "grid_shared"),
]


def run_all_models(
    videos_dir: Path | str = "videos",
    base_out_dir: Path | str = "videos_out",
    models: list[tuple[str, str]] | None = None,
    device: str | None = None,
) -> dict[str, list[Path]]:
    """Run inference for each (ckpt_dir, arch_name); ckpt = ckpt_dir/best.pt. Returns {arch_name: [output paths]}."""
    videos_dir = Path(videos_dir)
    base_out_dir = Path(base_out_dir)
    models = models or DEFAULT_MODELS
    results = {}
    for ckpt_dir, arch_name in models:
        ckpt_path = Path(ckpt_dir) / "best.pt"
        if not ckpt_path.exists():
            print(f"Skipping {arch_name}: {ckpt_path} not found")
            continue
        out_dir = base_out_dir / arch_name
        results[arch_name] = run_inference_for_model(
            ckpt_path, arch_name, videos_dir, out_dir, device=device, verbose=True
        )
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run multi-object detection on videos")
    p.add_argument("--videos_dir", type=Path, default=Path("videos"), help="Input videos folder")
    p.add_argument("--out_dir", type=Path, default=Path("videos_out"), help="Base output folder (per-model subdirs)")
    p.add_argument("--model", type=str, default=None, help="Run only this arch (default: all in DEFAULT_MODELS)")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    if args.model:
        ckpt_dir = f"checkpoints/part3_{args.model}"
        run_inference_for_model(
            Path(ckpt_dir) / "best.pt",
            args.model,
            args.videos_dir,
            args.out_dir / args.model,
            device=args.device,
        )
    else:
        run_all_models(videos_dir=args.videos_dir, base_out_dir=args.out_dir, device=args.device)
