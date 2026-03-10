"""
Training utilities for Project 3 — Part 3 (multi-object detection, fixed 3 slots).

Two loss modes (see loss_mode in configs):
- **fixed_slot**: Slot i is always matched to GT slot i (slot 0=helmet, 1=person, 2=vest).
  Bbox loss on non-background slots; CE on all slots. Simple and stable.
- **hungarian**: Set prediction — we find an optimal 1-to-1 assignment between the 3 predictions
  and the 3 GT slots (including background), then compute loss on matched pairs. No fixed slot
  semantics; model can output "person + helmet" without being forced to fill vest. Requires scipy.

Metrics: mean IoU over non-background slots (with fixed-slot alignment, or with matching when loss_mode=hungarian).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2 as T

from dataset import BACKGROUND_CLASS_ID, MultiObjectDataset
from model import DetectorConfig, MultiObjectDetector, NUM_CLASSES, NUM_SLOTS

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None  # Hungarian loss unavailable without scipy

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


LossMode = Literal["fixed_slot", "hungarian"]


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for mini-overfit debug."""

    split: str = "train"
    subset_size: int = 10
    batch_size: int = 10
    epochs: int = 200
    lr: float = 1e-3
    bbox_loss_weight: float = 1.0
    # Optional IoU-aware term in the loss: total = bbox*λ + class + iou_weight*(1 - mIoU)
    iou_loss_weight: float = 0.0
    loss_mode: LossMode = "fixed_slot"
    seed: int = 0
    num_workers: int = 0
    freeze_backbone: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class FullTrainConfig:
    """Configuration for full training with validation + TensorBoard."""

    train_split: Literal["train"] = "train"
    val_split: Literal["valid"] = "valid"
    batch_size: int = 16
    num_workers: int = 0

    model_cfg: DetectorConfig = field(default_factory=DetectorConfig)
    epochs_max: int = 60
    head_only_epochs: int = 10
    lr_head: float = 1e-3
    lr_layer4: float = 1e-4
    weight_decay: float = 3e-4
    bbox_loss_weight: float = 1.0
    loss_mode: LossMode = "fixed_slot"
    # Optional IoU-aware term in the loss: total = bbox*λ + class + iou_weight*(1 - mIoU)
    iou_loss_weight: float = 0.0

    use_plateau_scheduler: bool = True
    plateau_patience: int = 5
    plateau_factor: float = 0.5
    plateau_min_lr: float = 1e-6

    use_early_stopping: bool = False
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    logdir: str = "logs/part3_full"
    ckpt_dir: str = "checkpoints/part3"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_eval_transform() -> T.Compose:
    """Preprocessing for eval: ToImage, ToDtype, ImageNet normalize."""
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def make_train_transform() -> T.Compose:
    """Preprocessing for train: color jitter + ImageNet normalize (no geometric; bbox unchanged)."""
    return T.Compose([
        T.ToImage(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------- Loss and IoU ----------


def _cxcywh_to_xyxy_norm(b: torch.Tensor) -> torch.Tensor:
    """(..., 4) normalized (cx,cy,w,h) -> (..., 4) normalized (x1,y1,x2,y2)."""
    cx, cy, w, h = b.unbind(dim=-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _iou_xyxy_norm(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """pred, tgt (N,4) normalized xyxy -> (N,) IoU."""
    x1 = torch.maximum(pred[:, 0], tgt[:, 0])
    y1 = torch.maximum(pred[:, 1], tgt[:, 1])
    x2 = torch.minimum(pred[:, 2], tgt[:, 2])
    y2 = torch.minimum(pred[:, 3], tgt[:, 3])
    inter_w = (x2 - x1).clamp(min=0.0)
    inter_h = (y2 - y1).clamp(min=0.0)
    inter = inter_w * inter_h
    pred_a = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    tgt_a = (tgt[:, 2] - tgt[:, 0]).clamp(min=0) * (tgt[:, 3] - tgt[:, 1]).clamp(min=0)
    union = pred_a + tgt_a - inter
    return inter / (union + eps)


def compute_loss(
    pred_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_class_ids: torch.Tensor,
    *,
    bbox_loss_weight: float = 1.0,
    iou_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, float, float]:
    """
    Bbox loss only on non-background slots; CE on all slots.

    Returns
    -------
    total_loss, bbox_loss_scalar, class_loss_scalar (for logging).
    """
    # pred_boxes (B,3,4), pred_logits (B,3,4), gt_boxes (B,3,4), gt_class_ids (B,3) long
    mask = (gt_class_ids != BACKGROUND_CLASS_ID)  # (B, 3)
    n_bbox = mask.sum().clamp(min=1)

    bbox_loss = F.smooth_l1_loss(
        pred_boxes[mask],
        gt_boxes[mask],
        reduction="sum",
    ) / n_bbox

    class_loss = F.cross_entropy(
        pred_logits.view(-1, NUM_CLASSES),
        gt_class_ids.view(-1),
        reduction="mean",
    )

    total = bbox_loss_weight * bbox_loss + class_loss

    # Optional IoU-aware term: encourage higher mIoU directly.
    # We reuse mean_iou_multi so that the IoU term is computed over the same foreground slots
    # used for evaluation. When iou_loss_weight=0 (default), this is a no-op.
    if float(iou_loss_weight) > 0.0:
        iou_mean = mean_iou_multi(pred_boxes, gt_boxes, gt_class_ids)
        total = total + float(iou_loss_weight) * (1.0 - iou_mean)

    return total, float(bbox_loss.detach().cpu()), float(class_loss.detach().cpu())


def mean_iou_multi(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_class_ids: torch.Tensor,
) -> torch.Tensor:
    """Mean IoU over non-background slots only. pred/gt (B,3,4) normalized cxcywh."""
    mask = (gt_class_ids != BACKGROUND_CLASS_ID)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_boxes.device, dtype=pred_boxes.dtype)

    pred_xyxy = _cxcywh_to_xyxy_norm(pred_boxes).clamp(0.0, 1.0)
    gt_xyxy = _cxcywh_to_xyxy_norm(gt_boxes).clamp(0.0, 1.0)
    pred_flat = pred_xyxy.view(-1, 4)[mask.view(-1)]
    gt_flat = gt_xyxy.view(-1, 4)[mask.view(-1)]
    return _iou_xyxy_norm(pred_flat, gt_flat).mean()


def _hungarian_assign(
    pred_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_class_ids: torch.Tensor,
    bbox_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each sample in the batch, compute optimal assignment pred_slot -> gt_slot via cost matrix,
    then reorder gt so that gt_matched[b,i] = gt[b, col_ind[i]] where col_ind comes from the assignment.
    Returns (gt_boxes_matched, gt_class_matched) of same shapes as gt_boxes, gt_class_ids.
    """
    if linear_sum_assignment is None:
        raise RuntimeError("Hungarian matching requires scipy. Install with: pip install scipy")
    B = pred_boxes.shape[0]
    device = pred_boxes.device
    gt_boxes_matched = torch.zeros_like(gt_boxes, device=device)
    gt_class_matched = torch.full_like(gt_class_ids, BACKGROUND_CLASS_ID, device=device)

    for b in range(B):
        # Cost[i,j] = cost of assigning prediction i to gt slot j (minimize total)
        C = torch.zeros(NUM_SLOTS, NUM_SLOTS, device=device)
        for i in range(NUM_SLOTS):
            for j in range(NUM_SLOTS):
                cls_cost = F.cross_entropy(
                    pred_logits[b, i : i + 1],
                    gt_class_ids[b, j : j + 1].long(),
                    reduction="none",
                ).squeeze()
                if gt_class_ids[b, j].item() == BACKGROUND_CLASS_ID:
                    C[i, j] = cls_cost
                else:
                    box_cost = F.smooth_l1_loss(
                        pred_boxes[b, i : i + 1],
                        gt_boxes[b, j : j + 1],
                        reduction="sum",
                    )
                    C[i, j] = cls_cost + bbox_loss_weight * box_cost

        row_ind, col_ind = linear_sum_assignment(C.detach().cpu().numpy())
        # row_ind[k] = pred index, col_ind[k] = gt index; so pred[row_ind[k]] is matched to gt[col_ind[k]]
        # We want gt_matched[i] = gt[col_ind[inv[i]]] where inv[row_ind[k]] = k, i.e. inv[i] = k s.t. row_ind[k]==i
        inv = torch.zeros(NUM_SLOTS, dtype=torch.long, device=device)
        for k in range(NUM_SLOTS):
            inv[row_ind[k]] = k
        for i in range(NUM_SLOTS):
            j = col_ind[inv[i].item()]
            gt_boxes_matched[b, i] = gt_boxes[b, j]
            gt_class_matched[b, i] = gt_class_ids[b, j]

    return gt_boxes_matched, gt_class_matched


def compute_loss_hungarian(
    pred_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_class_ids: torch.Tensor,
    *,
    bbox_loss_weight: float = 1.0,
    iou_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, float, float]:
    """
    Set-prediction loss: optimal 1-to-1 assignment between 3 predictions and 3 GT slots (incl. background),
    then same per-slot loss as fixed_slot on the matched pairs. Differentiable only through pred after assignment.
    """
    gt_boxes_matched, gt_class_matched = _hungarian_assign(
        pred_boxes, pred_logits, gt_boxes, gt_class_ids, bbox_loss_weight
    )
    return compute_loss(
        pred_boxes,
        pred_logits,
        gt_boxes_matched,
        gt_class_matched,
        bbox_loss_weight=bbox_loss_weight,
        iou_loss_weight=iou_loss_weight,
    )


def mean_iou_multi_with_matching(
    pred_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_class_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Mean IoU after optimally matching predictions to GT (same cost as Hungarian loss).
    Use when loss_mode=hungarian for a consistent metric.
    """
    if linear_sum_assignment is None:
        return mean_iou_multi(pred_boxes, gt_boxes, gt_class_ids)
    gt_boxes_matched, gt_class_matched = _hungarian_assign(
        pred_boxes, pred_logits, gt_boxes, gt_class_ids, bbox_loss_weight=1.0
    )
    return mean_iou_multi(pred_boxes, gt_boxes_matched, gt_class_matched)


# ---------- Test set evaluation ----------


def evaluate_on_test(
    ckpt_path: Path | str,
    data_root: Path | None = None,
    *,
    batch_size: int = 16,
    device: str | None = None,
    loss_mode: LossMode = "fixed_slot",
) -> dict[str, float]:
    """
    Load checkpoint, run on test split, return test_miou and test_loss.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = data_root or Path(__file__).resolve().parent / "data"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("cfg") or {}
    model_cfg = cfg_dict.get("model_cfg")
    if model_cfg is None:
        model_cfg = DetectorConfig(arch="conv_shared", output_activation="sigmoid")
    elif isinstance(model_cfg, dict):
        allowed = {"weights", "arch", "output_activation", "mlp_dims", "mlp_dropouts", "conv_drop2d"}
        model_cfg = DetectorConfig(**{k: v for k, v in model_cfg.items() if k in allowed})

    model = MultiObjectDetector(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    test_ds = MultiObjectDataset(split="test", data_root=data_root, allow_empty=True, transform=make_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    total_loss = 0.0
    total_iou = 0.0
    n_batches = 0
    with torch.inference_mode():
        for imgs, gt_boxes, gt_class_ids in test_loader:
            imgs = imgs.to(device)
            gt_boxes = gt_boxes.to(device)
            gt_class_ids = gt_class_ids.to(device)
            pred_boxes, pred_logits = model(imgs)
            # For test_loss we keep the original SmoothL1+CE formulation (no IoU term)
            loss, _, _ = compute_loss(
                pred_boxes,
                pred_logits,
                gt_boxes,
                gt_class_ids,
                bbox_loss_weight=1.0,
                iou_loss_weight=0.0,
            )
            if loss_mode == "hungarian":
                miou = mean_iou_multi_with_matching(pred_boxes, pred_logits, gt_boxes, gt_class_ids)
            else:
                miou = mean_iou_multi(pred_boxes, gt_boxes, gt_class_ids)
            total_loss += float(loss.cpu())
            total_iou += float(miou.cpu())
            n_batches += 1
    n_batches = max(1, n_batches)
    return {"test_loss": total_loss / n_batches, "test_miou": total_iou / n_batches}


# ---------- DataLoaders ----------


def build_subset_loader(
    cfg: TrainConfig,
    data_root: Path | None = None,
    indices: list[int] | None = None,
) -> DataLoader:
    """DataLoader over a small subset for mini-overfit."""
    ds = MultiObjectDataset(
        split=cfg.split,
        data_root=data_root,
        allow_empty=True,
        transform=make_eval_transform(),
    )
    set_seed(cfg.seed)
    if indices is None:
        indices = list(range(len(ds)))
        random.shuffle(indices)
        indices = indices[: cfg.subset_size]
    subset = Subset(ds, indices)
    return DataLoader(
        subset,
        batch_size=min(cfg.batch_size, len(indices)),
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
    )


# ---------- Mini-overfit ----------


def mini_overfit_with_tensorboard(
    *,
    writer: SummaryWriter,
    model_cfg: DetectorConfig | None = None,
    subset_size: int = 10,
    batch_size: int = 10,
    epochs: int = 200,
    lr: float = 1e-3,
    bbox_loss_weight: float = 1.0,
    loss_mode: LossMode = "fixed_slot",
    weight_decay: float = 3e-4,
    seed: int = 0,
    indices: list[int] | None = None,
    freeze_backbone: bool = False,
    device: str | None = None,
    data_root: Path | None = None,
) -> None:
    """
    Mini-overfit debug: train on a tiny subset, log loss and mIoU to TensorBoard.

    Expectation: loss decreases, mIoU increases toward 1. If not, check data/model/loss wiring.
    loss_mode: "fixed_slot" (slot i = gt i) or "hungarian" (set prediction with optimal matching).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    loader = build_subset_loader(
        TrainConfig(
            subset_size=subset_size,
            batch_size=batch_size,
            seed=seed,
            device=device,
            loss_mode=loss_mode,
        ),
        data_root=data_root,
        indices=indices,
    )

    model = MultiObjectDetector(model_cfg or DetectorConfig(weights="DEFAULT", output_activation="sigmoid")).to(device)
    if freeze_backbone:
        model.freeze_backbone()

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )

    global_step = 0
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_bbox = 0.0
        total_class = 0.0
        total_iou = 0.0
        n_batches = 0

        for imgs, gt_boxes, gt_class_ids in loader:
            imgs = imgs.to(device)
            gt_boxes = gt_boxes.to(device)
            gt_class_ids = gt_class_ids.to(device)

            pred_boxes, pred_logits = model(imgs)
            if loss_mode == "hungarian":
                loss, bbox_s, class_s = compute_loss_hungarian(
                    pred_boxes,
                    pred_logits,
                    gt_boxes,
                    gt_class_ids,
                    bbox_loss_weight=bbox_loss_weight,
                )
            else:
                loss, bbox_s, class_s = compute_loss(
                    pred_boxes,
                    pred_logits,
                    gt_boxes,
                    gt_class_ids,
                    bbox_loss_weight=bbox_loss_weight,
                )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            with torch.no_grad():
                if loss_mode == "hungarian":
                    miou = mean_iou_multi_with_matching(pred_boxes, pred_logits, gt_boxes, gt_class_ids)
                else:
                    miou = mean_iou_multi(pred_boxes, gt_boxes, gt_class_ids)

            writer.add_scalar("train/loss_step", float(loss.detach().cpu()), global_step)
            writer.add_scalar("train/mIoU_step", float(miou.cpu()), global_step)
            writer.add_scalar("train/bbox_loss_step", bbox_s, global_step)
            writer.add_scalar("train/class_loss_step", class_s, global_step)
            global_step += 1

            total_loss += float(loss.detach().cpu())
            total_bbox += bbox_s
            total_class += class_s
            total_iou += float(miou.cpu())
            n_batches += 1

        n_batches = max(1, n_batches)
        avg_loss = total_loss / n_batches
        avg_bbox = total_bbox / n_batches
        avg_class = total_class / n_batches
        avg_iou = total_iou / n_batches

        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        writer.add_scalar("train/mIoU_epoch", avg_iou, epoch)
        writer.add_scalar("train/bbox_loss_epoch", avg_bbox, epoch)
        writer.add_scalar("train/class_loss_epoch", avg_class, epoch)

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"epoch={epoch:03d}  loss={avg_loss:.6f}  bbox={avg_bbox:.6f}  class={avg_class:.6f}  mIoU={avg_iou:.4f}")


# ---------- Full training (stub: run_one_epoch + train_full) ----------


def _run_one_epoch(
    *,
    model: MultiObjectDetector,
    loader: DataLoader,
    device: str,
    optim: torch.optim.Optimizer | None,
    cfg: FullTrainConfig,
) -> tuple[float, float, float, float]:
    """One epoch; returns avg_loss, avg_bbox_loss, avg_class_loss, avg_mIoU."""
    is_train = optim is not None
    model.train(is_train)

    total_loss = 0.0
    total_bbox = 0.0
    total_class = 0.0
    total_iou = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.inference_mode()
    with ctx:
        for imgs, gt_boxes, gt_class_ids in loader:
            imgs = imgs.to(device)
            gt_boxes = gt_boxes.to(device)
            gt_class_ids = gt_class_ids.to(device)

            pred_boxes, pred_logits = model(imgs)
            if cfg.loss_mode == "hungarian":
                loss, bbox_s, class_s = compute_loss_hungarian(
                    pred_boxes,
                    pred_logits,
                    gt_boxes,
                    gt_class_ids,
                    bbox_loss_weight=cfg.bbox_loss_weight,
                    iou_loss_weight=cfg.iou_loss_weight,
                )
            else:
                loss, bbox_s, class_s = compute_loss(
                    pred_boxes,
                    pred_logits,
                    gt_boxes,
                    gt_class_ids,
                    bbox_loss_weight=cfg.bbox_loss_weight,
                    iou_loss_weight=cfg.iou_loss_weight,
                )

            if is_train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

            if cfg.loss_mode == "hungarian":
                miou = mean_iou_multi_with_matching(pred_boxes, pred_logits, gt_boxes, gt_class_ids)
            else:
                miou = mean_iou_multi(pred_boxes, gt_boxes, gt_class_ids)

            total_loss += float(loss.detach().cpu())
            total_bbox += bbox_s
            total_class += class_s
            total_iou += float(miou.cpu())
            n_batches += 1

    n_batches = max(1, n_batches)
    return (
        total_loss / n_batches,
        total_bbox / n_batches,
        total_class / n_batches,
        total_iou / n_batches,
    )


def _build_phase_b_optimizer(model: MultiObjectDetector, cfg: FullTrainConfig) -> torch.optim.Optimizer:
    head_params = list(model.head.parameters())
    layer4_params = list(model.backbone.layer4.parameters())
    return torch.optim.AdamW(
        [
            {"params": layer4_params, "lr": cfg.lr_layer4, "weight_decay": cfg.weight_decay},
            {"params": head_params, "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
        ]
    )


def _save_checkpoint(
    path: Path,
    model: MultiObjectDetector,
    optim: torch.optim.Optimizer,
    cfg: FullTrainConfig,
    epoch: int,
    best_val_miou: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "best_val_miou": best_val_miou,
            "cfg": {k: getattr(cfg, k) for k in ["model_cfg", "epochs_max", "head_only_epochs", "lr_head", "lr_layer4", "weight_decay", "bbox_loss_weight", "loss_mode"]},
        },
        path,
    )


def train_full(cfg: FullTrainConfig, data_root: Path | None = None) -> None:
    """
    Full training: Phase A (head only), Phase B (unfreeze layer4), validation, TensorBoard, checkpoints.
    """
    set_seed(cfg.seed)
    device = cfg.device

    train_ds = MultiObjectDataset(
        split=cfg.train_split,
        data_root=data_root,
        allow_empty=True,
        transform=make_train_transform(),
    )
    val_ds = MultiObjectDataset(
        split=cfg.val_split,
        data_root=data_root,
        allow_empty=True,
        transform=make_eval_transform(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    model = MultiObjectDetector(cfg.model_cfg).to(device)
    model.freeze_backbone()
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    scheduler = None
    if cfg.use_plateau_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="max", patience=cfg.plateau_patience, factor=cfg.plateau_factor, min_lr=cfg.plateau_min_lr
        )

    logdir = Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(logdir))
    ckpt_dir = Path(cfg.ckpt_dir)
    best_val_miou = -1.0
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs_max + 1):
        if epoch == cfg.head_only_epochs + 1:
            model.unfreeze_layer4()
            optim = _build_phase_b_optimizer(model, cfg)
            if cfg.use_plateau_scheduler:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optim, mode="max", patience=cfg.plateau_patience, factor=cfg.plateau_factor, min_lr=cfg.plateau_min_lr
                )

        train_loss, train_bbox, train_class, train_miou = _run_one_epoch(
            model=model, loader=train_loader, device=device, optim=optim, cfg=cfg
        )
        val_loss, val_bbox, val_class, val_miou = _run_one_epoch(
            model=model, loader=val_loader, device=device, optim=None, cfg=cfg
        )

        lrs = [g["lr"] for g in optim.param_groups]
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("train/mIoU_epoch", train_miou, epoch)
        writer.add_scalar("train/bbox_loss_epoch", train_bbox, epoch)
        writer.add_scalar("train/class_loss_epoch", train_class, epoch)
        writer.add_scalar("val/loss_epoch", val_loss, epoch)
        writer.add_scalar("val/mIoU_epoch", val_miou, epoch)
        writer.add_scalar("val/bbox_loss_epoch", val_bbox, epoch)
        writer.add_scalar("val/class_loss_epoch", val_class, epoch)
        writer.add_scalar("optim/lr_group0", lrs[0], epoch)
        if len(lrs) > 1:
            writer.add_scalar("optim/lr_group1", lrs[1], epoch)

        _save_checkpoint(ckpt_dir / "last.pt", model, optim, cfg, epoch, best_val_miou)
        if val_miou > best_val_miou + cfg.early_stop_min_delta:
            best_val_miou = val_miou
            _save_checkpoint(ckpt_dir / "best.pt", model, optim, cfg, epoch, best_val_miou)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if scheduler is not None:
            scheduler.step(val_miou)

        if cfg.use_early_stopping and epochs_no_improve >= cfg.early_stop_patience:
            print(f"Early stopping after {epoch} epochs.")
            break

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs_max:
            print(f"epoch={epoch:03d}  train_loss={train_loss:.6f} train_mIoU={train_miou:.4f}  val_loss={val_loss:.6f} val_mIoU={val_miou:.4f}")

    writer.close()
    print(f"Best val mIoU: {best_val_miou:.4f}")
