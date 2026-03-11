"""
Model definitions for Project 3 - Part 2 (single-class, single-object detection).

We adapt a pretrained ResNet18 backbone into a bounding-box regressor.

Core assumptions (kept consistent across architecture variants):
- Single object per image
- Output format: normalized (cx, cy, w, h) in [0, 1]
- Output activation: sigmoid (by default)

This matches the target representation produced by `SafetyVestDataset` in `dataset.py`.

Architecture variants supported (see `DetectorConfig.arch`):
1) MLP head with per-layer dropout (baseline, stable)
2) Conv head before pooling (preserves spatial information)
3) Two-head MLP (center + size branches)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import torch
import torch.nn as nn
from torchvision import models


WeightsName = Literal["DEFAULT", "IMAGENET1K_V1", "none"]
ArchName = Literal["mlp_simple", "mlp", "conv_head", "two_head"]
OutputActivation = Literal["sigmoid", "none"]


@dataclass(frozen=True)
class DetectorConfig:
    """
    Configuration for `SingleObjectDetector` (architecture + hyperparameters).

    Attributes
    ----------
    weights:
        Which ResNet18 weights to load. Use "DEFAULT" for torchvision's default ImageNet weights.
        Use "none" to initialize randomly (not recommended for this project).
    arch:
        Which head architecture to use. See module docstring for details.
    output_activation:
        Activation used to map raw outputs to [0, 1]. "sigmoid" is the common choice when targets
        are normalized. "none" returns unconstrained outputs (you'd then clamp or use a different
        parametrization in training).
    mlp_dims:
        Hidden sizes for the MLP head. Example: (256, 128) builds 512 -> 256 -> 128 -> 4.
    mlp_dropouts:
        Dropout probabilities after each hidden layer (same length as `mlp_dims`).
    conv_drop2d:
        Dropout2d probability for the conv-head variant.
    """

    weights: WeightsName = "DEFAULT"
    arch: ArchName = "mlp"
    output_activation: OutputActivation = "sigmoid"

    # Architecture 1 (MLP head)
    mlp_dims: tuple[int, ...] = (256, 128)
    mlp_dropouts: tuple[float, ...] = (0.4, 0.3)

    # Architecture 2 (Conv-head)
    conv_drop2d: float = 0.2

    # Architecture 0 (simple MLP, like the original baseline)
    simple_hidden_dim: int = 128


class SingleObjectDetector(nn.Module):
    """
    A single-object detector that regresses one bbox per image.

    Returns normalized (cx, cy, w, h). By default we apply sigmoid to keep outputs in [0,1].
    """

    def __init__(self, cfg: DetectorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DetectorConfig()

        # --- Backbone ---
        if self.cfg.weights == "none":
            weights = None
        elif self.cfg.weights == "IMAGENET1K_V1":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = models.ResNet18_Weights.DEFAULT

        self.backbone = models.resnet18(weights=weights)

        # We'll use the ResNet blocks explicitly to get feature maps (B,512,H,W).
        # This keeps the model flexible for conv-head variants.
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

        # --- Head(s) ---
        self.head = self._build_head()

        if self.cfg.output_activation == "sigmoid":
            self.out_act: nn.Module = nn.Sigmoid()
        else:
            self.out_act = nn.Identity()

    def _build_head(self) -> nn.Module:
        arch = self.cfg.arch

        if arch == "mlp_simple":
            return _MLPHead(
                in_dim=512,
                hidden_dims=(int(self.cfg.simple_hidden_dim),),
                dropouts=(0.0,),
                out_dim=4,
            )

        if arch == "mlp":
            return _MLPHead(
                in_dim=512,
                hidden_dims=self.cfg.mlp_dims,
                dropouts=self.cfg.mlp_dropouts,
                out_dim=4,
            )

        if arch == "conv_head":
            return _ConvHead(
                in_ch=512,
                drop2d=float(self.cfg.conv_drop2d),
                out_dim=4,
            )

        if arch == "two_head":
            # Shared trunk: 512 -> 256 (+dropout), then separate branches.
            # Dropouts are taken from mlp_dropouts[0] if provided.
            shared_drop = float(self.cfg.mlp_dropouts[0]) if len(self.cfg.mlp_dropouts) > 0 else 0.4
            return _TwoHead(
                in_dim=512,
                shared_dim=256,
                branch_dim=128,
                shared_dropout=shared_drop,
            )

        raise ValueError(f"Unknown arch: {arch}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the last conv feature map (B, 512, H, W)."""
        return self._stem(x)

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled features (B, 512)."""
        f = self.forward_features(x)
        f = self._pool(f).flatten(1)
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Image batch tensor of shape (B, 3, H, W), float.

        Returns
        -------
        torch.Tensor
            Bounding boxes of shape (B, 4) in normalized (cx, cy, w, h).
        """
        if self.cfg.arch == "conv_head":
            raw = self.head(self.forward_features(x))  # (B,4)
        else:
            raw = self.head(self.forward_pooled(x))  # (B,4)
        return self.out_act(raw)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters; keep head trainable."""
        for p in self._stem.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_layer4(self) -> None:
        """Unfreeze only ResNet layer4 + head (common fine-tuning recipe)."""
        self.freeze_backbone()
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze all model parameters."""
        for p in self.parameters():
            p.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _MLPHead(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dims: Sequence[int],
        dropouts: Sequence[float],
        out_dim: int,
    ) -> None:
        super().__init__()

        if len(hidden_dims) != len(dropouts):
            raise ValueError("mlp_dims and mlp_dropouts must have the same length")

        layers: list[nn.Module] = []
        prev = int(in_dim)
        for h, d in zip(hidden_dims, dropouts):
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU(inplace=True))
            if float(d) > 0.0:
                layers.append(nn.Dropout(p=float(d)))
            prev = int(h)

        layers.append(nn.Linear(prev, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ConvHead(nn.Module):
    def __init__(self, *, in_ch: int, drop2d: float, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=float(drop2d)) if float(drop2d) > 0.0 else nn.Identity(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=float(drop2d)) if float(drop2d) > 0.0 else nn.Identity(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, int(out_dim))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.conv(feat)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class _TwoHead(nn.Module):
    def __init__(self, *, in_dim: int, shared_dim: int, branch_dim: int, shared_dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(int(in_dim), int(shared_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(shared_dropout)) if float(shared_dropout) > 0.0 else nn.Identity(),
        )
        self.center = nn.Sequential(
            nn.Linear(int(shared_dim), int(branch_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(branch_dim), 2),
        )
        self.size = nn.Sequential(
            nn.Linear(int(shared_dim), int(branch_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(branch_dim), 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        c = self.center(h)
        s = self.size(h)
        return torch.cat([c, s], dim=-1)

