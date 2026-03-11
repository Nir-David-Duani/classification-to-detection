"""
Training utilities for Project 3 — Part 2 (single-object bbox regression).

Stage 4 (mini-overfit) is implemented here as a debug routine:
- Train on a tiny subset (e.g., 5 images) with no geometric augmentation.
- Expectation: loss -> ~0 and IoU -> ~1.

Why this matters:
If mini-overfit fails, there is almost certainly a bug in:
  - target bbox representation
  - model output representation
  - loss wiring / normalization
  - data/model device placement
  - IoU implementation
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2 as T

from dataset import SafetyVestDataset
from model import DetectorConfig, SingleObjectDetector

# ImageNet normalization statistics expected by torchvision pretrained backbones.
# Reference: https://pytorch.org/vision/stable/models.html#classification
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for the mini-overfit debug experiment."""

    split: str = "train"
    subset_size: int = 5
    batch_size: int = 5
    epochs: int = 200
    lr: float = 1e-3
    seed: int = 0
    num_workers: int = 0
    pin_memory: bool = True
    freeze_backbone: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass(frozen=True)
class FullTrainConfig:
    """
    Configuration for Stage 5 (full training with validation + TensorBoard).

    Notes
    -----
    - This config uses *non-geometric* augmentations by default, so bbox targets remain valid.
    - For geometric augmentations (flip/crop/resize), you must update the bbox accordingly.
    """

    # Data
    train_split: Literal["train"] = "train"
    val_split: Literal["valid"] = "valid"
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True

    # Model
    model_cfg: DetectorConfig = field(default_factory=DetectorConfig)

    # Optimization (fixed assumptions for fair comparisons)
    epochs_max: int = 60
    head_only_epochs: int = 10  # Phase A: freeze backbone, train head only
    lr_head: float = 1e-3
    lr_layer4: float = 1e-4
    weight_decay: float = 3e-4  # AdamW default in this project

    # LR scheduling
    use_plateau_scheduler: bool = True
    plateau_patience: int = 5
    plateau_factor: float = 0.5
    plateau_min_lr: float = 1e-6

    # Early stopping (based on val mIoU)
    use_early_stopping: bool = False
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4

    # Repro / device
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging / outputs
    logdir: str = "logs/part2_full"
    ckpt_dir: str = "checkpoints/part2"

    # Loss options
    loss_mode: Literal["smoothl1", "two_head_weighted", "smoothl1_plus_iou"] = "smoothl1"
    two_head_center_w: float = 0.6
    two_head_size_w: float = 0.4
    iou_loss_w: float = 0.3  # final loss = (1-iou_loss_w)*SmoothL1 + iou_loss_w*(1-mIoU)


def set_seed(seed: int) -> None:
    """Set seeds for reproducible subset sampling and training."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cxcywh_to_xyxy_norm(b: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized (cx,cy,w,h) to normalized (x1,y1,x2,y2).

    Parameters
    ----------
    b:
        Tensor of shape (..., 4) in normalized (cx,cy,w,h).
    """
    cx, cy, w, h = b.unbind(dim=-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def clamp_xyxy01(xyxy: torch.Tensor) -> torch.Tensor:
    """Clamp normalized xyxy coordinates to [0,1]."""
    return xyxy.clamp(0.0, 1.0)


def iou_xyxy_norm(pred_xyxy: torch.Tensor, tgt_xyxy: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute IoU between normalized xyxy boxes.

    Parameters
    ----------
    pred_xyxy, tgt_xyxy:
        Tensors of shape (B,4) with coords in [0,1].
    """
    # Intersection
    x1 = torch.maximum(pred_xyxy[:, 0], tgt_xyxy[:, 0])
    y1 = torch.maximum(pred_xyxy[:, 1], tgt_xyxy[:, 1])
    x2 = torch.minimum(pred_xyxy[:, 2], tgt_xyxy[:, 2])
    y2 = torch.minimum(pred_xyxy[:, 3], tgt_xyxy[:, 3])

    inter_w = (x2 - x1).clamp(min=0.0)
    inter_h = (y2 - y1).clamp(min=0.0)
    inter = inter_w * inter_h

    # Areas
    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0.0) * (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0.0)
    tgt_area = (tgt_xyxy[:, 2] - tgt_xyxy[:, 0]).clamp(min=0.0) * (tgt_xyxy[:, 3] - tgt_xyxy[:, 1]).clamp(min=0.0)
    union = pred_area + tgt_area - inter

    return inter / (union + eps)


def mean_iou(pred_cxcywh: torch.Tensor, tgt_cxcywh: torch.Tensor) -> torch.Tensor:
    """Compute mean IoU for a batch of normalized (cx,cy,w,h) boxes."""
    pred_xyxy = clamp_xyxy01(cxcywh_to_xyxy_norm(pred_cxcywh))
    tgt_xyxy = clamp_xyxy01(cxcywh_to_xyxy_norm(tgt_cxcywh))
    return iou_xyxy_norm(pred_xyxy, tgt_xyxy).mean()


def compute_loss(cfg: FullTrainConfig, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the training loss according to cfg.loss_mode.

    preds/targets are normalized cxcywh in [0,1] (after sigmoid in the model).
    """

    if cfg.loss_mode == "smoothl1":
        return nn.functional.smooth_l1_loss(preds, targets)

    if cfg.loss_mode == "two_head_weighted":
        center = nn.functional.smooth_l1_loss(preds[:, 0:2], targets[:, 0:2])
        size = nn.functional.smooth_l1_loss(preds[:, 2:4], targets[:, 2:4])
        return float(cfg.two_head_center_w) * center + float(cfg.two_head_size_w) * size

    if cfg.loss_mode == "smoothl1_plus_iou":
        base = nn.functional.smooth_l1_loss(preds, targets)
        miou = mean_iou(preds, targets)
        iou_term = 1.0 - miou
        w = float(cfg.iou_loss_w)
        return (1.0 - w) * base + w * iou_term

    raise ValueError(f"Unknown loss_mode: {cfg.loss_mode}")


def build_subset_loader(cfg: TrainConfig) -> DataLoader:
    """Create a DataLoader over a small random subset of the dataset."""
    # No augmentation for mini-overfit, but keep preprocessing consistent with the pretrained backbone.
    ds = SafetyVestDataset(split=cfg.split, sample_transform=make_eval_sample_transform())
    set_seed(cfg.seed)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[: cfg.subset_size]

    subset = Subset(ds, indices)
    return DataLoader(
        subset,
        batch_size=min(cfg.batch_size, cfg.subset_size),
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and (cfg.device.startswith("cuda")),
    )

def make_train_transform() -> Any:
    """
    Non-geometric augmentations (safe for bbox regression without bbox updates).

    Returns float32 tensors normalized with ImageNet mean/std, shape (3,H,W).
    """
    return T.Compose(
        [
            T.ToImage(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            # Make blur probabilistic; always blurring can hurt localization.
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_eval_transform() -> Any:
    """Evaluation preprocessing (no augmentation)."""
    return T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_train_sample_transform(*, flip_p: float = 0.5) -> Any:
    """
    BBox-aware train transform for (PIL image, bbox_cxcywh_norm) -> (image_tensor, bbox_tensor).

    This keeps augmentation safe by updating the bbox for geometric transforms.
    Currently supported geometric augmentation:
      - Horizontal flip (updates cx: cx' = 1 - cx)
    """

    img_tf = make_train_transform()

    def _tf(img, bbox: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bbox_out = bbox

        # Apply horizontal flip with bbox update (cx only, since bbox is normalized cxcywh).
        if flip_p > 0.0 and torch.rand(()) < float(flip_p):
            bbox_out = bbox.clone()
            bbox_out[0] = 1.0 - bbox_out[0]

            # Convert + augment + normalize, then flip the tensor horizontally (last dim).
            img_t = img_tf(img)
            img_t = img_t.flip(-1)
            return img_t, bbox_out

        # No flip: just image augmentations + normalization.
        return img_tf(img), bbox_out

    return _tf


def make_eval_sample_transform() -> Any:
    """BBox-preserving eval transform for (PIL image, bbox) -> (image_tensor, bbox_tensor)."""
    img_tf = make_eval_transform()

    def _tf(img, bbox: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return img_tf(img), bbox

    return _tf


def _set_finetune_scope(model: SingleObjectDetector, scope: Literal["layer4", "all"]) -> None:
    """
    Control which backbone parameters are trainable for fine-tuning.

    - layer4: unfreeze only the last ResNet stage (layer4) + regression head
    - all: unfreeze all parameters
    """
    if scope == "all":
        model.unfreeze_all()
        return

    # Start from fully frozen backbone, then selectively unfreeze.
    model.freeze_backbone()

    # Unfreeze layer4 specifically.
    for p in model.backbone.layer4.parameters():
        p.requires_grad = True

    # Ensure head is trainable.
    for p in model.backbone.fc.parameters():
        p.requires_grad = True


def _save_checkpoint(path: Path, model: nn.Module, optim: torch.optim.Optimizer, cfg: FullTrainConfig, epoch: int, best_val_miou: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "best_val_miou": best_val_miou,
            "cfg": cfg.__dict__,
        },
        path,
    )


def _run_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optim: torch.optim.Optimizer | None,
    cfg: FullTrainConfig,
) -> tuple[float, float]:
    """
    Run one epoch of training or evaluation.

    If optim is None -> evaluation (no backward, no step).
    Returns: (avg_loss, avg_mIoU).
    """
    is_train = optim is not None
    model.train(is_train)

    total_loss = 0.0
    total_iou = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.inference_mode()
    with ctx:
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            preds = model(imgs)
            loss = compute_loss(cfg, preds, targets)

            if is_train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

            miou = mean_iou(preds, targets)

            total_loss += float(loss.detach().cpu())
            total_iou += float(miou.detach().cpu())
            n_batches += 1

    return (total_loss / max(1, n_batches), total_iou / max(1, n_batches))


def _build_adamw(params: Iterable[torch.nn.Parameter], *, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))


def _build_phase_b_optimizer(model: SingleObjectDetector, cfg: FullTrainConfig) -> torch.optim.Optimizer:
    """
    Phase B optimizer with two parameter groups:
    - layer4: lr_layer4
    - head:   lr_head
    """

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    layer4_params = [p for p in model.backbone.layer4.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": layer4_params, "lr": float(cfg.lr_layer4), "weight_decay": float(cfg.weight_decay)},
            {"params": head_params, "lr": float(cfg.lr_head), "weight_decay": float(cfg.weight_decay)},
        ]
    )


def _maybe_make_scheduler(cfg: FullTrainConfig, optim: torch.optim.Optimizer):
    if not cfg.use_plateau_scheduler:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="max",
        patience=int(cfg.plateau_patience),
        factor=float(cfg.plateau_factor),
        min_lr=float(cfg.plateau_min_lr),
    )


def train_full(cfg: FullTrainConfig) -> None:
    """
    Stage 5: Full training with validation + TensorBoard + checkpoints.

    Logs (epoch-level):
      - train/loss_epoch, train/mIoU_epoch
      - val/loss_epoch, val/mIoU_epoch
      - lr
      - trainable_params
    """
    set_seed(cfg.seed)
    device = cfg.device

    train_ds = SafetyVestDataset(split=cfg.train_split, sample_transform=make_train_sample_transform())
    val_ds = SafetyVestDataset(split=cfg.val_split, sample_transform=make_eval_sample_transform())

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.startswith("cuda"),
    )

    model = SingleObjectDetector(cfg.model_cfg).to(device)

    # Phase A — head only
    model.freeze_backbone()
    optim = _build_adamw((p for p in model.parameters() if p.requires_grad), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    scheduler = _maybe_make_scheduler(cfg, optim)

    logdir = Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(logdir))

    best_val_miou = -1.0
    epochs_no_improve = 0
    ckpt_dir = Path(cfg.ckpt_dir)

    for epoch in range(1, cfg.epochs_max + 1):
        # Phase B — unfreeze layer4
        if epoch == cfg.head_only_epochs + 1:
            model.unfreeze_layer4()
            optim = _build_phase_b_optimizer(model, cfg)
            scheduler = _maybe_make_scheduler(cfg, optim)

        train_loss, train_miou = _run_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optim=optim,
            cfg=cfg,
        )
        val_loss, val_miou = _run_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optim=None,
            cfg=cfg,
        )

        # Logging
        lrs = [float(g["lr"]) for g in optim.param_groups]
        lr0 = lrs[0] if lrs else float("nan")
        lr1 = lrs[1] if len(lrs) > 1 else lr0
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("train/mIoU_epoch", train_miou, epoch)
        writer.add_scalar("val/loss_epoch", val_loss, epoch)
        writer.add_scalar("val/mIoU_epoch", val_miou, epoch)
        writer.add_scalar("optim/lr_group0", lr0, epoch)
        writer.add_scalar("optim/lr_group1", lr1, epoch)
        writer.add_scalar("model/trainable_params", sum(p.numel() for p in model.parameters() if p.requires_grad), epoch)
        writer.add_scalar("cfg/head_only_epochs", float(cfg.head_only_epochs), epoch)

        # Checkpoints
        _save_checkpoint(ckpt_dir / "last.pt", model, optim, cfg, epoch, best_val_miou)
        improved = val_miou > (best_val_miou + float(cfg.early_stop_min_delta))
        if improved:
            best_val_miou = val_miou
            _save_checkpoint(ckpt_dir / "best.pt", model, optim, cfg, epoch, best_val_miou)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Scheduler step (validation metric)
        if scheduler is not None:
            scheduler.step(val_miou)

        # Early stopping (optional)
        if cfg.use_early_stopping and epochs_no_improve >= int(cfg.early_stop_patience):
            print(f"Early stopping: no val mIoU improvement for {epochs_no_improve} epochs.")
            break

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs_max:
            print(
                f"epoch={epoch:03d}  "
                f"train_loss={train_loss:.6f} train_mIoU={train_miou:.4f}  "
                f"val_loss={val_loss:.6f} val_mIoU={val_miou:.4f}  "
                f"lr0={lr0:.2e} lr1={lr1:.2e}"
            )

    writer.flush()
    writer.close()
    print(f"Best val mIoU: {best_val_miou:.4f}")


def train_mini_overfit(cfg: TrainConfig) -> None:
    """
    Run the stage-4 mini-overfit experiment.

    Prints training loss and IoU over epochs. Success criteria:
      - loss approaches ~0
      - IoU approaches ~1
    """
    loader = build_subset_loader(cfg)

    model = SingleObjectDetector(DetectorConfig(weights="DEFAULT", output_activation="sigmoid")).to(cfg.device)
    if cfg.freeze_backbone:
        model.freeze_backbone()

    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=3e-4)
    loss_fn: nn.Module = nn.SmoothL1Loss()

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        total_iou = 0.0
        n_batches = 0

        for imgs, targets in loader:
            imgs = imgs.to(cfg.device)
            targets = targets.to(cfg.device)

            preds = model(imgs)
            loss = loss_fn(preds, targets)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            with torch.no_grad():
                miou = mean_iou(preds, targets)

            total_loss += float(loss.detach().cpu())
            total_iou += float(miou.detach().cpu())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_iou = total_iou / max(1, n_batches)

        if epoch == 1 or epoch % 20 == 0 or epoch == cfg.epochs:
            print(f"epoch={epoch:03d}  loss={avg_loss:.6f}  mIoU={avg_iou:.4f}")


def mini_overfit_with_tensorboard(
    *,
    writer: SummaryWriter,
    model_cfg: DetectorConfig | None = None,
    loss_mode: Literal["smoothl1", "two_head_weighted", "smoothl1_plus_iou"] = "smoothl1",
    two_head_center_w: float = 0.6,
    two_head_size_w: float = 0.4,
    iou_loss_w: float = 0.3,
    subset_size: int = 5,
    batch_size: int = 5,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 3e-4,
    seed: int = 0,
    indices: list[int] | None = None,
    freeze_backbone: bool = False,
    device: str | None = None,
) -> None:
    """
    Mini-overfit debug training with TensorBoard logging.

    This is the same Stage-4 idea as `train_mini_overfit`, but logs metrics live:
    - train/loss_step, train/mIoU_step
    - train/loss_epoch, train/mIoU_epoch

    Parameters
    ----------
    writer:
        TensorBoard SummaryWriter that determines the logdir/run.
    subset_size:
        Number of training images to sample from the train split.
    batch_size:
        Batch size (usually equals subset_size for the cleanest debug run).
    epochs:
        Number of epochs.
    lr:
        Adam learning rate.
    seed:
        Random seed used for subset sampling and training reproducibility.
    freeze_backbone:
        If True, freeze the ResNet backbone and train only the head.
    device:
        If None, auto-selects 'cuda' if available, else 'cpu'.
    """

    """
    Mini-overfit debug training with TensorBoard logging.

    This is intended as a *sanity check* when experimenting with new architectures/losses.
    Expectation on a tiny subset:
      - loss -> ~0
      - mIoU -> ~1

    Parameters
    ----------
    model_cfg:
        `DetectorConfig` describing the architecture under test.
    loss_mode:
        Which loss to use. For architecture sweeps, keep the loss consistent unless you are
        explicitly testing an IoU-aware objective.
    indices:
        Optional fixed subset indices to make mini-overfit runs comparable across architectures.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    # No augmentation for mini-overfit, but keep preprocessing consistent with the pretrained backbone.
    ds = SafetyVestDataset(split="train", sample_transform=make_eval_sample_transform())
    if indices is None:
        indices = list(range(len(ds)))
        random.shuffle(indices)
        indices = indices[:subset_size]

    loader = DataLoader(
        Subset(ds, indices),
        batch_size=min(batch_size, subset_size),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.startswith("cuda")),
    )

    cfg = FullTrainConfig(
        model_cfg=model_cfg or DetectorConfig(weights="DEFAULT", output_activation="sigmoid"),
        loss_mode=loss_mode,
        two_head_center_w=two_head_center_w,
        two_head_size_w=two_head_size_w,
        iou_loss_w=iou_loss_w,
        weight_decay=weight_decay,
    )

    model = SingleObjectDetector(cfg.model_cfg).to(device)
    if freeze_backbone:
        model.freeze_backbone()

    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)

    global_step = 0
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_iou = 0.0
        n_batches = 0

        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            preds = model(imgs)
            loss = compute_loss(cfg, preds, targets)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            with torch.no_grad():
                miou = mean_iou(preds, targets)

            # step-level logging
            writer.add_scalar("train/loss_step", float(loss.detach().cpu()), global_step)
            writer.add_scalar("train/mIoU_step", float(miou.detach().cpu()), global_step)
            global_step += 1

            total_loss += float(loss.detach().cpu())
            total_iou += float(miou.detach().cpu())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_iou = total_iou / max(1, n_batches)

        # epoch-level logging
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        writer.add_scalar("train/mIoU_epoch", avg_iou, epoch)

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"epoch={epoch:03d}  loss={avg_loss:.6f}  mIoU={avg_iou:.4f}")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Project 3 Part 2 — mini-overfit debug training")
    p.add_argument("--split", default="train", choices=["train", "valid", "test"])
    p.add_argument("--subset-size", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--freeze-backbone", action="store_true")
    args = p.parse_args()
    return TrainConfig(
        split=args.split,
        subset_size=args.subset_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        freeze_backbone=bool(args.freeze_backbone),
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_mini_overfit(cfg)

