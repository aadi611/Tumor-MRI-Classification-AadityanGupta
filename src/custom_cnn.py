"""
custom_cnn.py
=============
A *from-scratch* 2-D CNN for brain-tumor MRI classification — NO transfer
learning, NO pretrained weights. This is our own architecture, designed to be
competitive with the ImageNet backbones in the portfolio while remaining fully
interpretable layer-by-layer.

Design choices (the "advanced techniques" the brief asked for):
  * Residual connections      — ease gradient flow, let us go deep without decay.
  * Squeeze-and-Excitation     — lightweight channel attention; the network learns
    (SE) attention blocks         *which feature maps matter* per image.
  * Batch normalisation        — stabilises and speeds up training of deep nets.
  * Spatial + head dropout     — regularises a relatively small medical dataset.
  * SiLU (Swish) activations   — smoother than ReLU and consistently better on
                                  image classification (used by EfficientNet).
  * Kaiming initialisation     — essential from scratch so the signal neither
                                  vanishes nor explodes at depth (the ReLU gain
                                  is the standard approximation for SiLU).

Architecture at a glance (input assumed 3x224x224):

    Stem  : Conv7x7/2 -> BN -> SiLU -> MaxPool3x3/2          ->  64 x 56 x 56
    Stage1: 2 x ResidualSE(64  -> 64 )  stride 1             ->  64 x 56 x 56
    Stage2: 2 x ResidualSE(64  -> 128)  stride 2 (first)     -> 128 x 28 x 28
    Stage3: 2 x ResidualSE(128 -> 256)  stride 2 (first)     -> 256 x 14 x 14
    Stage4: 2 x ResidualSE(256 -> 512)  stride 2 (first)     -> 512 x  7 x  7
    Head  : GlobalAvgPool -> Dropout -> Linear(512 -> 4)     ->   4 (logits)

The whole thing is ~11 M parameters — comparable to ResNet-18 — and outputs
raw logits (softmax is applied by the loss / at inference, never inside forward).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────
# Squeeze-and-Excitation channel attention
# ──────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (Hu et al., 2018).

    Recalibrates channel responses: 'squeeze' global spatial info into a per-
    channel descriptor, then 'excite' a learned gate in [0, 1] per channel and
    rescale the feature map. Cheap (two small FC layers) but consistently helpful.
    """

    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        hidden = max(channels // ratio, 4)          # bottleneck width
        self.pool = nn.AdaptiveAvgPool2d(1)         # squeeze: (B,C,H,W) -> (B,C,1,1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape                        # x: (B, C, H, W)
        s = self.pool(x).view(b, c)                 # squeeze -> (B, C)
        s = self.fc(s).view(b, c, 1, 1)             # excite  -> (B, C, 1, 1) gate in [0,1]
        return x * s                                # channel-wise reweight, shape unchanged


# ──────────────────────────────────────────────────────────────────────────
# Residual block with SE attention
# ──────────────────────────────────────────────────────────────────────────
class ResidualSEBlock(nn.Module):
    """Two 3x3 convs + SE, wrapped in a residual (skip) connection.

    out = SiLU( SE(conv-bn-silu-conv-bn(x)) + shortcut(x) )

    When the spatial size or channel count changes (stride>1 or in!=out) the
    identity shortcut is replaced by a 1x1 projection so the add stays valid.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 dropout: float = 0.0, se_ratio: int = 16):
        super().__init__()
        # Main path -----------------------------------------------------------
        # conv1 carries the stride -> may downsample H,W:  (B,in,H,W)->(B,out,H/s,W/s)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        # conv2 keeps spatial size:                        (B,out,H/s,W/s)->(B,out,H/s,W/s)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch, se_ratio)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        # SiLU (Swish): smooth, non-monotonic — consistently edges out ReLU
        # on image classification (it's what EfficientNet uses throughout).
        self.act = nn.SiLU(inplace=True)

        # Skip path -----------------------------------------------------------
        # Identity if shapes match, else a 1x1 conv projection to align them.
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),  # (B,in,H,W)->(B,out,H/s,W/s)
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)                 # (B, out, H/s, W/s)
        out = self.act(self.bn1(self.conv1(x)))     # (B, out, H/s, W/s)
        out = self.bn2(self.conv2(out))             # (B, out, H/s, W/s)
        out = self.se(out)                          # channel-attention reweight
        out = self.drop(out)
        out = self.act(out + identity)              # residual add + non-linearity
        return out


# ──────────────────────────────────────────────────────────────────────────
# The full network
# ──────────────────────────────────────────────────────────────────────────
class CustomBrainCNN(nn.Module):
    """From-scratch residual SE-CNN for 4-class brain-tumor MRI classification.

    Parameters
    ----------
    num_classes : number of output logits (4 here).
    dropout     : dropout probability for the classifier head.
    width       : channel width of the 4 stages.
    blocks      : number of ResidualSE blocks per stage.
    block_drop  : light spatial dropout inside residual blocks (deep-layer reg.).
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.4,
                 width: Sequence[int] = (64, 128, 256, 512),
                 blocks: Sequence[int] = (2, 2, 2, 2),
                 block_drop: float = 0.1):
        super().__init__()

        # ── Stem: aggressive early downsampling, like ResNet ────────────────
        # (B, 3, 224, 224) -> Conv7x7/2 -> (B, 64, 112, 112) -> MaxPool/2 -> (B, 64, 56, 56)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # -> (B,64,112,112)
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                  # -> (B,64,56,56)
        )

        # ── Residual SE stages ──────────────────────────────────────────────
        # Stage 1 keeps 56x56 (stride 1); stages 2-4 each halve the resolution.
        self.stage1 = self._make_stage(64,       width[0], blocks[0], stride=1, drop=block_drop)  # (B,64,56,56)
        self.stage2 = self._make_stage(width[0], width[1], blocks[1], stride=2, drop=block_drop)  # (B,128,28,28)
        self.stage3 = self._make_stage(width[1], width[2], blocks[2], stride=2, drop=block_drop)  # (B,256,14,14)
        self.stage4 = self._make_stage(width[2], width[3], blocks[3], stride=2, drop=block_drop)  # (B,512,7,7)

        # ── Classifier head ─────────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)          # (B,512,7,7) -> (B,512,1,1)
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width[-1], num_classes)   # (B,512) -> (B,num_classes)

        self._init_weights()

    # Build a stage: first block may downsample / change channels, rest are 1:1.
    def _make_stage(self, in_ch: int, out_ch: int, n_blocks: int,
                    stride: int, drop: float) -> nn.Sequential:
        layers = [ResidualSEBlock(in_ch, out_ch, stride=stride, dropout=drop)]
        for _ in range(n_blocks - 1):
            layers.append(ResidualSEBlock(out_ch, out_ch, stride=1, dropout=drop))
        return nn.Sequential(*layers)

    # Proper initialisation is critical when there are no pretrained weights.
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nonlinearity="relu" stays: SiLU has no dedicated gain in
                # PyTorch and the ReLU gain is the standard approximation.
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def arch_summary(self) -> None:
        """Print a parameter-count / memory-footprint summary of the network."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size_mb = total * 4 / (1024 ** 2)   # float32 = 4 bytes per param
        print(f"Total parameters   : {total:,}")
        print(f"Trainable params   : {trainable:,}")
        print(f"Estimated size     : {size_mb:.1f} MB  (float32, params only)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224)
        x = self.stem(x)                # -> (B,  64, 56, 56)
        x = self.stage1(x)              # -> (B,  64, 56, 56)
        x = self.stage2(x)              # -> (B, 128, 28, 28)
        x = self.stage3(x)              # -> (B, 256, 14, 14)
        x = self.stage4(x)              # -> (B, 512,  7,  7)
        x = self.gap(x).flatten(1)      # -> (B, 512)
        x = self.head_dropout(x)
        return self.classifier(x)       # -> (B, num_classes)  raw logits
