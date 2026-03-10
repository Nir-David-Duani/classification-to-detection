"""
Model definitions for Project 3 — Part 3 (multi-class, multi-object detection, fixed capacity).

We adapt a pretrained ResNet18 backbone into a detector that outputs a fixed number of slots
per image. Each slot predicts one bounding box (normalized cx, cy, w, h) and one of four
classes: helmet, person, vest, or background (no object).

Core assumptions (kept consistent across architecture variants):
- At most 3 objects per image; slots are ordered: 0=helmet, 1=person, 2=vest.
- Output format: pred_boxes (B, 3, 4) in normalized (cx, cy, w, h) in [0, 1]; sigmoid by default.
- pred_logits (B, 3, 4): class logits per slot (helmet, person, vest, background); no softmax here.
- This matches the target representation from `MultiObjectDataset` in `dataset.py`.

Architecture variants supported (see `DetectorConfig.arch`):
0) mlp_shared — MLP on pooled backbone features → two Linear branches (boxes 12, logits 12); reshape to (B,3,4).
1) conv_shared — Conv head on feature map (same as Part 2 best) → 128-d → two Linear (12, 12); reshape.
2) conv_per_slot — Same conv head → 128-d → per-slot Linear layers (3× boxes 4, 3× logits 4); stack to (B,3,4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


WeightsName = Literal["DEFAULT", "IMAGENET1K_V1", "none"]
ArchName = Literal[
    "mlp_shared",
    "conv_shared",
    "conv_shared_deep",
    "conv_per_slot",
    "conv_per_slot_l3",
    "conv_fpn",
    "grid_shared",
]
OutputActivation = Literal["sigmoid", "none"]

NUM_SLOTS = 3
NUM_CLASSES = 4  # helmet, person, vest, background


@dataclass(frozen=True)
class DetectorConfig:
    """
    Configuration for `MultiObjectDetector` (architecture + hyperparameters).

    Attributes
    ----------
    weights:
        Which ResNet18 weights to load. "DEFAULT" for torchvision ImageNet weights.
    arch:
        Which head architecture to use. See module docstring for details.
    output_activation:
        Activation for bbox outputs: "sigmoid" keeps (cx,cy,w,h) in [0,1]. "none" leaves raw.
    mlp_dims:
        Hidden sizes for the MLP head (arch=mlp_shared). Example: (256, 128) → 512→256→128.
    mlp_dropouts:
        Dropout after each hidden layer in MLP head (same length as mlp_dims).
    conv_drop2d:
        Dropout2d probability for conv-head variants.
    """

    weights: WeightsName = "DEFAULT"
    arch: ArchName = "conv_shared"
    output_activation: OutputActivation = "sigmoid"

    mlp_dims: tuple[int, ...] = (256, 128)
    mlp_dropouts: tuple[float, ...] = (0.4, 0.3)
    conv_drop2d: float = 0.2


class MultiObjectDetector(nn.Module):
    """
    Multi-object detector: fixed 3 slots per image, each slot predicts bbox + 4 class logits.

    Returns
    -------
    pred_boxes : (B, 3, 4) float, normalized (cx, cy, w, h) in [0, 1] when output_activation=sigmoid.
    pred_logits : (B, 3, 4) float, class logits (helmet, person, vest, background); no softmax.
    """

    def __init__(self, cfg: DetectorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DetectorConfig()

        if self.cfg.weights == "none":
            weights = None
        elif self.cfg.weights == "IMAGENET1K_V1":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = models.ResNet18_Weights.DEFAULT

        self.backbone = models.resnet18(weights=weights)
        # Keep a stem that goes all the way to layer4 for existing architectures.
        # For some variants we will explicitly tap into layer3 (higher spatial resolution).
        self._stem = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        )
        self._pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = self._build_head()
        self.out_act = nn.Sigmoid() if self.cfg.output_activation == "sigmoid" else nn.Identity()

    def _build_head(self) -> nn.Module:
        arch = self.cfg.arch
        if arch == "mlp_shared":
            return _MLPSharedHead(
                in_dim=512,
                hidden_dims=self.cfg.mlp_dims,
                dropouts=self.cfg.mlp_dropouts,
                num_slots=NUM_SLOTS,
                num_classes=NUM_CLASSES,
            )
        if arch == "conv_shared":
            return _ConvSharedHead(
                in_ch=512,
                drop2d=float(self.cfg.conv_drop2d),
                num_slots=NUM_SLOTS,
                num_classes=NUM_CLASSES,
            )
        if arch == "conv_shared_deep":
            return _ConvSharedDeepHead(
                in_ch=512,
                drop2d=float(self.cfg.conv_drop2d),
                num_slots=NUM_SLOTS,
                num_classes=NUM_CLASSES,
            )
        if arch == "conv_fpn":
            return _FPNSharedHead(
                drop2d=float(self.cfg.conv_drop2d),
                num_slots=NUM_SLOTS,
                num_classes=NUM_CLASSES,
            )
        if arch == "grid_shared":
            return _GridSharedHead(
                in_ch=512,
                drop2d=float(self.cfg.conv_drop2d),
                num_slots=NUM_SLOTS,
                num_classes=NUM_CLASSES,
            )
        if arch == "conv_per_slot":
            return _ConvPerSlotHead(
                in_ch=512,
                drop2d=float(self.cfg.conv_drop2d),
                num_slots=NUM_SLOTS,
                num_classes=NUM_CLASSES,
            )
        if arch == "conv_per_slot_l3":
            # Variant that uses layer3 features (14×14, 256 channels) instead of layer4 (7×7, 512).
            return _ConvPerSlotHead(
                in_ch=256,
                drop2d=float(self.cfg.conv_drop2d),
                num_slots=NUM_SLOTS,
                num_classes=NUM_CLASSES,
            )
        raise ValueError(f"Unknown arch: {arch}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the last conv feature map (B, 512, H, W)."""
        return self._stem(x)

    def forward_features_l3(self, x: torch.Tensor) -> torch.Tensor:
        """Return conv feature map after layer3 (B, 256, 14, 14 for 224×224 input)."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        return x

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled features (B, 512)."""
        return self._pool(self.forward_features(x)).flatten(1)

    def forward_pooled_l3(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled features from layer3 (B, 256) — kept generic for possible future heads."""
        return self._pool(self.forward_features_l3(x)).flatten(1)

    def forward_features_fpn(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return feature maps from layer2, layer3, layer4 for a small FPN:
        C2: (B, 128, 28, 28), C3: (B, 256, 14, 14), C4: (B, 512, 7, 7) for 224×224 input.
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        c2 = x
        x = self.backbone.layer3(x)
        c3 = x
        x = self.backbone.layer4(x)
        c4 = x
        return c2, c3, c4

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 3, H, W) float

        Returns
        -------
        pred_boxes : (B, 3, 4) in normalized (cx, cy, w, h).
        pred_logits : (B, 3, 4) class logits per slot.
        """
        if self.cfg.arch == "conv_per_slot_l3":
            # Use higher-resolution layer3 features for this architecture.
            return self.head(
                x,
                self.forward_features_l3,
                self.forward_pooled_l3,
                self.out_act,
            )
        if self.cfg.arch == "conv_fpn":
            # Use multiple backbone levels (layer2, layer3, layer4) fused by a small FPN head.
            return self.head(
                x,
                self.forward_features_fpn,
                self.forward_pooled,
                self.out_act,
            )
        return self.head(x, self.forward_features, self.forward_pooled, self.out_act)

    def freeze_backbone(self) -> None:
        for p in self._stem.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_layer4(self) -> None:
        self.freeze_backbone()
        if self.cfg.arch == "conv_per_slot_l3":
            # For the layer3-based architecture, fine-tune the last backbone block we actually use.
            for p in self.backbone.layer3.parameters():
                p.requires_grad = True
        else:
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----- Head implementations -----


class _MLPSharedHead(nn.Module):
    """Pooled 512-d → MLP → two branches: boxes (12) and logits (12); reshape to (B,3,4)."""

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dims: Sequence[int],
        dropouts: Sequence[float],
        num_slots: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_classes = num_classes
        if len(hidden_dims) != len(dropouts):
            raise ValueError("hidden_dims and dropouts must have same length")

        layers: list[nn.Module] = []
        prev = in_dim
        for h, d in zip(hidden_dims, dropouts):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if float(d) > 0:
                layers.append(nn.Dropout(p=float(d)))
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.fc_boxes = nn.Linear(prev, num_slots * 4)
        self.fc_logits = nn.Linear(prev, num_slots * num_classes)

    def forward(
        self,
        x: torch.Tensor,
        forward_features: nn.Module,
        forward_pooled: nn.Module,
        out_act: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = forward_pooled(x)
        h = self.mlp(h)
        boxes = out_act(self.fc_boxes(h)).view(-1, self.num_slots, 4)
        logits = self.fc_logits(h).view(-1, self.num_slots, self.num_classes)
        return boxes, logits


class _ConvSharedHead(nn.Module):
    """Conv head on feature map (1×1, 3×3, pool) → 128-d → two Linear(12); reshape (B,3,4)."""

    def __init__(
        self,
        *,
        in_ch: int,
        drop2d: float,
        num_slots: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 256, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop2d) if drop2d > 0 else nn.Identity(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop2d) if drop2d > 0 else nn.Identity(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_boxes = nn.Linear(128, num_slots * 4)
        self.fc_logits = nn.Linear(128, num_slots * num_classes)

    def forward(
        self,
        x: torch.Tensor,
        forward_features: nn.Module,
        forward_pooled: nn.Module,
        out_act: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = forward_features(x)
        h = self.pool(self.conv(feat)).flatten(1)
        boxes = out_act(self.fc_boxes(h)).view(-1, self.num_slots, 4)
        logits = self.fc_logits(h).view(-1, self.num_slots, self.num_classes)
        return boxes, logits


class _ConvSharedDeepHead(nn.Module):
    """
    Deeper conv head on feature map: 3×(Conv+ReLU+Dropout2d) → pool → 128-d → two Linear(12).
    Increases capacity for localization while keeping shared head.
    """

    def __init__(
        self,
        *,
        in_ch: int,
        drop2d: float,
        num_slots: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop2d) if drop2d > 0 else nn.Identity(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop2d) if drop2d > 0 else nn.Identity(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop2d) if drop2d > 0 else nn.Identity(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_boxes = nn.Linear(128, num_slots * 4)
        self.fc_logits = nn.Linear(128, num_slots * num_classes)

    def forward(
        self,
        x: torch.Tensor,
        forward_features: nn.Module,
        forward_pooled: nn.Module,
        out_act: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = forward_features(x)
        h = self.pool(self.conv(feat)).flatten(1)
        boxes = out_act(self.fc_boxes(h)).view(-1, self.num_slots, 4)
        logits = self.fc_logits(h).view(-1, self.num_slots, self.num_classes)
        return boxes, logits


class _FPNSharedHead(nn.Module):
    """
    Small FPN-style head that fuses layer2, layer3, and layer4 feature maps
    (C2, C3, C4) and predicts 3 slots × (bbox, class logits).
    """

    def __init__(
        self,
        *,
        drop2d: float,
        num_slots: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_classes = num_classes

        # Lateral 1×1 convs to 128 channels from C2 (128ch), C3 (256ch), C4 (512ch).
        self.lat2 = nn.Conv2d(128, 128, kernel_size=1)
        self.lat3 = nn.Conv2d(256, 128, kernel_size=1)
        self.lat4 = nn.Conv2d(512, 128, kernel_size=1)

        # 3×3 smoothing convs on each pyramid level.
        self.smooth2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=drop2d) if drop2d > 0 else nn.Identity()

        # Fuse pooled P2, P3, P4 into a 128-d vector, then predict boxes/logits.
        self.fc_fuse = nn.Linear(128 * 3, 128)
        self.fc_boxes = nn.Linear(128, num_slots * 4)
        self.fc_logits = nn.Linear(128, num_slots * num_classes)

    def forward(
        self,
        x: torch.Tensor,
        forward_features: nn.Module,
        forward_pooled: nn.Module,
        out_act: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # forward_features returns (C2, C3, C4).
        c2, c3, c4 = forward_features(x)

        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)

        g2 = self.pool(p2).flatten(1)
        g3 = self.pool(p3).flatten(1)
        g4 = self.pool(p4).flatten(1)

        h = torch.cat([g2, g3, g4], dim=1)
        h = self.dropout(F.relu(self.fc_fuse(h)))

        boxes = out_act(self.fc_boxes(h)).view(-1, self.num_slots, 4)
        logits = self.fc_logits(h).view(-1, self.num_slots, self.num_classes)
        return boxes, logits


class _GridSharedHead(nn.Module):
    """
    Grid-based head (YOLO-style) on a 7×7 feature map.

    Internally predicts per-cell (objectness, bbox, class logits) for each slot
    over a 7×7 grid, then aggregates them into 3 slots (helmet, person, vest)
    via a softmax-weighted average over cells. This keeps the external interface
    the same: (B, 3, 4) boxes and (B, 3, 4) logits.
    """

    def __init__(
        self,
        *,
        in_ch: int,
        drop2d: float,
        num_slots: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_classes = num_classes

        # Simple conv tower to produce a 7×7 grid of per-cell predictions.
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop2d) if drop2d > 0 else nn.Identity(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop2d) if drop2d > 0 else nn.Identity(),
        )

        # For each slot and each cell we predict:
        # - 4 bbox params (cx, cy, w, h) in normalized space
        # - num_classes class logits
        # - 1 objectness logit
        out_channels = num_slots * (4 + num_classes + 1)
        self.pred = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        forward_features: nn.Module,
        forward_pooled: nn.Module,
        out_act: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Backbone features after layer4: (B, in_ch, 7, 7) for 224×224 input.
        feat = forward_features(x)
        h = self.conv(feat)
        b, _, h_h, h_w = h.shape  # expect 7×7 but keep it generic

        pred = self.pred(h)  # (B, num_slots*(4+num_classes+1), H, W)
        pred = pred.view(
            b,
            self.num_slots,
            4 + self.num_classes + 1,
            h_h,
            h_w,
        )

        bbox_raw = pred[:, :, 0:4, :, :]  # (B, S, 4, H, W)
        cls_raw = pred[:, :, 4 : 4 + self.num_classes, :, :]  # (B, S, C, H, W)
        obj_raw = pred[:, :, 4 + self.num_classes, :, :]  # (B, S, H, W)

        # Softmax over grid cells per slot → weights for a soft selection of the cell.
        grid_size = h_h * h_w
        obj_flat = obj_raw.view(b, self.num_slots, grid_size)  # (B, S, HW)
        weights = F.softmax(obj_flat, dim=-1)  # (B, S, HW)

        # Aggregate bbox: weighted average over cells (after sigmoid to keep [0,1]).
        # weights (B,S,HW), bbox_flat (B,S,4,HW) -> sum over last dim -> (B,S,4)
        bbox_flat = out_act(bbox_raw).view(b, self.num_slots, 4, grid_size)  # (B,S,4,HW)
        bbox = torch.einsum("bsk,bsck->bsc", weights, bbox_flat)  # (B,S,4)

        # Aggregate class logits: weighted average over cells.
        cls_flat = cls_raw.view(b, self.num_slots, self.num_classes, grid_size)  # (B,S,C,HW)
        logits = torch.einsum("bsk,bsck->bsc", weights, cls_flat)  # (B,S,C)

        return bbox, logits


class _ConvPerSlotHead(nn.Module):
    """Same conv head → 128-d; then per-slot: Linear(128,4) boxes and Linear(128,4) logits; stack (B,3,4)."""

    def __init__(
        self,
        *,
        in_ch: int,
        drop2d: float,
        num_slots: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 256, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop2d) if drop2d > 0 else nn.Identity(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop2d) if drop2d > 0 else nn.Identity(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_boxes = nn.ModuleList([nn.Linear(128, 4) for _ in range(num_slots)])
        self.fc_logits = nn.ModuleList([nn.Linear(128, num_classes) for _ in range(num_slots)])

    def forward(
        self,
        x: torch.Tensor,
        forward_features: nn.Module,
        forward_pooled: nn.Module,
        out_act: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = forward_features(x)
        h = self.pool(self.conv(feat)).flatten(1)  # (B, 128)
        boxes_list = [out_act(self.fc_boxes[i](h)) for i in range(self.num_slots)]
        logits_list = [self.fc_logits[i](h) for i in range(self.num_slots)]
        boxes = torch.stack(boxes_list, dim=1)   # (B, 3, 4)
        logits = torch.stack(logits_list, dim=1) # (B, 3, 4)
        return boxes, logits
