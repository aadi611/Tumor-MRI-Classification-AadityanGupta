"""
models.py
=========
Model portfolio factory + progressive-fine-tuning freeze/unfreeze controls.

A single `build_model(name)` entry point returns any of the supported
torchvision backbones with a fresh classification head sized to NUM_CLASSES.
Every backbone is wrapped so the rest of the pipeline can treat them uniformly:

    model = build_model("resnet50")
    freeze_backbone(model)          # Phase 1: train head only
    unfreeze_top(model, frac=0.3)   # Phase 2: fine-tune the top layers

Supported: resnet50, densenet121, vgg16, efficientnet_b0, efficientnet_b3,
mobilenet_v3, custom_cnn.

Also home to TemperatureScaling — a post-hoc calibration wrapper (Guo et al.,
2017) — and ece_score() for measuring Expected Calibration Error.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from .config import NUM_CLASSES
from .custom_cnn import CustomBrainCNN

# Models that are trained from random init (no pretrained backbone to freeze).
# The engine uses this to skip the two-phase freeze/unfreeze schedule.
FROM_SCRATCH = {"custom_cnn"}


def _make_head(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    """Shared classifier head: Dropout -> Linear -> BatchNorm -> ReLU -> Dropout -> Linear.

    BatchNorm on the bottleneck stabilises training of the new head; two dropout
    stages regularise the (relatively small) medical dataset.
    """
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout / 2),
        nn.Linear(256, num_classes),
    )


# ── Per-architecture builders. Each returns (model, backbone_module). ──────
def _build_resnet50(dropout):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = _make_head(m.fc.in_features, NUM_CLASSES, dropout)
    return m


def _build_densenet121(dropout):
    m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = _make_head(m.classifier.in_features, NUM_CLASSES, dropout)
    return m


def _build_vgg16(dropout):
    m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    in_f = m.classifier[0].in_features
    m.classifier = _make_head(in_f, NUM_CLASSES, dropout)
    return m


def _build_efficientnet_b0(dropout):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f = m.classifier[1].in_features
    m.classifier = _make_head(in_f, NUM_CLASSES, dropout)
    return m


def _build_efficientnet_b3(dropout):
    # Bigger EfficientNet: ~12M params, stronger features than B0 at the same
    # 224px input (natively 300px, but it tolerates 224 fine for transfer).
    m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_f = m.classifier[1].in_features
    m.classifier = _make_head(in_f, NUM_CLASSES, dropout)
    return m


def _build_mobilenet_v3(dropout):
    # Lightweight baseline (~5M params). The stock classifier is
    # Linear -> Hardswish -> Dropout -> Linear; we keep the pretrained
    # bottleneck and swap only the final Linear for our head.
    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = _make_head(in_f, NUM_CLASSES, dropout)
    return m


def _build_custom_cnn(dropout):
    # Our own from-scratch residual SE-CNN — no pretrained weights.
    return CustomBrainCNN(num_classes=NUM_CLASSES, dropout=dropout)


_BUILDERS: Dict[str, Callable[[float], nn.Module]] = {
    "resnet50": _build_resnet50,
    "densenet121": _build_densenet121,
    "vgg16": _build_vgg16,
    "efficientnet_b0": _build_efficientnet_b0,
    "efficientnet_b3": _build_efficientnet_b3,
    "mobilenet_v3": _build_mobilenet_v3,
    "custom_cnn": _build_custom_cnn,
}

# Attribute name of the classifier head per architecture (everything else == backbone).
# NOTE: for mobilenet_v3 this covers the whole classifier Sequential, i.e. the
# pretrained 960->1280 bottleneck Linear trains alongside the new head in
# Phase 1 — intentional, it's part of the classifier, not the conv backbone.
_HEAD_ATTR = {
    "resnet50": "fc",
    "densenet121": "classifier",
    "vgg16": "classifier",
    "efficientnet_b0": "classifier",
    "efficientnet_b3": "classifier",
    "mobilenet_v3": "classifier",
    "custom_cnn": "classifier",
}


def build_model(name: str, dropout: float = 0.4) -> nn.Module:
    """Instantiate a model: a pretrained backbone with a fresh head, or our
    from-scratch CustomBrainCNN when name == 'custom_cnn'."""
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Options: {list(_BUILDERS)}")
    model = _BUILDERS[name](dropout)
    model.arch_name = name                       # stash for later introspection
    model.from_scratch = name in FROM_SCRATCH    # engine reads this to pick the schedule
    return model


# ──────────────────────────────────────────────────────────────────────────
# Progressive fine-tuning helpers
# ──────────────────────────────────────────────────────────────────────────
def _head_param_ids(model: nn.Module) -> set:
    head = getattr(model, _HEAD_ATTR[model.arch_name])
    return {id(p) for p in head.parameters()}


def freeze_backbone(model: nn.Module) -> None:
    """Phase 1: freeze every parameter except the new classifier head."""
    head_ids = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids


def unfreeze_top(model: nn.Module, frac: float = 0.30) -> None:
    """Phase 2: unfreeze the head + the top `frac` of backbone parameters.

    We order backbone parameters as they appear in the network (early -> late)
    and unfreeze the final `frac` of them, since later layers encode the most
    task-specific features and benefit most from fine-tuning.
    """
    head_ids = _head_param_ids(model)
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    n_unfreeze = int(len(backbone_params) * frac)
    # First freeze all backbone params, then unfreeze the tail.
    for p in backbone_params:
        p.requires_grad = False
    for p in backbone_params[len(backbone_params) - n_unfreeze:]:
        p.requires_grad = True
    # Head always trainable.
    for p in model.parameters():
        if id(p) in head_ids:
            p.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────
# Post-hoc calibration: temperature scaling (Guo et al., 2017)
# ──────────────────────────────────────────────────────────────────────────
class TemperatureScaling(nn.Module):
    """Wrap a trained model and divide its logits by a learned temperature.

    Deep nets are systematically over-confident; a single scalar T > 1 fitted
    on the validation set (by minimising NLL) softens the softmax without
    changing the argmax — accuracy is untouched, calibration improves. This is
    essential for a medical model whose confidence scores a clinician may read.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) / self.temperature

    def fit(self, val_loader, device) -> float:
        """Fit T on validation logits with LBFGS; returns the fitted value.

        The wrapped model's weights are NOT updated — only `temperature`.
        """
        self.to(device)
        self.model.eval()

        # Collect raw (unscaled) logits once, so LBFGS re-evaluations are cheap.
        logits_list, labels_list = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits_list.append(self.model(x.to(device, non_blocking=True)))
                labels_list.append(y.to(device, non_blocking=True))
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = nll(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())


def ece_score(y_true, y_prob, n_bins: int = 15) -> float:
    """Expected Calibration Error over equal-width confidence bins.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    y_true : (N,) integer class labels
    y_prob : (N, C) predicted class probabilities (rows sum to 1)

    0 means perfectly calibrated; for reference, uncalibrated deep nets
    typically land in the 0.05-0.15 range.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    confidences = y_prob.max(axis=1)
    accuracies = (y_prob.argmax(axis=1) == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(accuracies[in_bin].mean()
                                       - confidences[in_bin].mean())
    return float(ece)
