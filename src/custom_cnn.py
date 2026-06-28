"""
custom_cnn.py
=============
From-scratch CNN for brain-tumor MRI — upgraded to reach 98–99% accuracy.

Changes vs. the original (11M / ResNet-18 scale):
  * Depth    : (2,2,2,2) blocks  →  (3,4,6,3)  — ResNet-50 depth
  * Attention: SE only           →  CBAM (channel SE + spatial gate)
  * Pooling  : GAP only          →  concat [GAP, GMP] → 2× features to head
  * ~25 M parameters (same channels as original; depth is the lever)

Architecture at a glance (3×256×256 input):

    Stem  : Conv7×7/2 → BN → SiLU → MaxPool3×3/2         →  64 × 64 × 64
    Stage1: 3 × ResidualCBAM( 64 →  64) stride 1          →  64 × 64 × 64
    Stage2: 4 × ResidualCBAM( 64 → 128) stride 2 (first)  → 128 × 32 × 32
    Stage3: 6 × ResidualCBAM(128 → 256) stride 2 (first)  → 256 × 16 × 16
    Stage4: 3 × ResidualCBAM(256 → 512) stride 2 (first)  → 512 ×  8 ×  8
    Head  : [GAP; GMP] → 1024 → Dropout → 512 → BN → SiLU → Dropout → 4
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────
# Channel attention (Squeeze-and-Excitation)
# ──────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        hidden = max(channels // ratio, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        return x * self.fc(s).view(b, c, 1, 1)


# ──────────────────────────────────────────────────────────────────────────
# Spatial attention gate (CBAM spatial branch)
# ──────────────────────────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """Pool across channels (avg + max), predict a per-pixel gate in [0,1]."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)          # (B,1,H,W)
        mx = x.max(dim=1, keepdim=True).values      # (B,1,H,W)
        gate = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * gate


# ──────────────────────────────────────────────────────────────────────────
# CBAM = channel SE + spatial attention
# ──────────────────────────────────────────────────────────────────────────
class CBAMBlock(nn.Module):
    def __init__(self, channels: int, se_ratio: int = 16, spatial_k: int = 7):
        super().__init__()
        self.se = SEBlock(channels, se_ratio)
        self.spatial = SpatialAttention(spatial_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.se(x))


# ──────────────────────────────────────────────────────────────────────────
# Residual block with CBAM attention
# ──────────────────────────────────────────────────────────────────────────
class ResidualCBAMBlock(nn.Module):
    """Two 3×3 convs + CBAM, wrapped in a residual connection.

    out = SiLU( CBAM( conv-bn-silu-conv-bn(x) ) + shortcut(x) )
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 dropout: float = 0.0, se_ratio: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,      padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.cbam  = CBAMBlock(out_ch, se_ratio)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act   = nn.SiLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out = self.drop(out)
        return self.act(out + identity)


# ──────────────────────────────────────────────────────────────────────────
# Full network
# ──────────────────────────────────────────────────────────────────────────
class CustomBrainCNN(nn.Module):
    """High-capacity residual CBAM-CNN for 4-class brain-tumor MRI.

    Parameters
    ----------
    num_classes : output logits (4 here)
    dropout     : head dropout probability
    width       : channel widths for the 4 stages
    blocks      : ResidualCBAMBlock count per stage
    block_drop  : spatial dropout inside residual blocks
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.4,
                 width: Sequence[int] = (64, 128, 256, 512),
                 blocks: Sequence[int] = (3, 4, 6, 3),
                 block_drop: float = 0.1):
        super().__init__()

        # Stem: (B,3,256,256) → (B,64,64,64)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = self._make_stage(64,       width[0], blocks[0], stride=1, drop=block_drop)
        self.stage2 = self._make_stage(width[0], width[1], blocks[1], stride=2, drop=block_drop)
        self.stage3 = self._make_stage(width[1], width[2], blocks[2], stride=2, drop=block_drop)
        self.stage4 = self._make_stage(width[2], width[3], blocks[3], stride=2, drop=block_drop)

        # Multi-scale pooling: concat GAP and GMP → 2× width[-1] features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        feat_dim = width[-1] * 2          # 1024
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _make_stage(self, in_ch: int, out_ch: int, n_blocks: int,
                    stride: int, drop: float) -> nn.Sequential:
        layers = [ResidualCBAMBlock(in_ch, out_ch, stride=stride, dropout=drop)]
        for _ in range(n_blocks - 1):
            layers.append(ResidualCBAMBlock(out_ch, out_ch, stride=1, dropout=drop))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def arch_summary(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters   : {total:,}")
        print(f"Trainable params   : {trainable:,}")
        print(f"Estimated size     : {total * 4 / 1024**2:.1f} MB  (float32)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # Multi-scale pooling
        x = torch.cat([self.gap(x).flatten(1),
                        self.gmp(x).flatten(1)], dim=1)
        return self.head(x)
