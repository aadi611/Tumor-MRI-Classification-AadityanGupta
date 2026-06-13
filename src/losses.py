"""
losses.py
=========
Loss functions + a small factory so the training code can swap criteria from config.

We support two losses, both fed RAW LOGITS (never softmax — that happens inside):

  * CrossEntropyLoss  — the proper, well-calibrated default. Optionally weighted by
    inverse class frequency and with label smoothing. This is our baseline.
  * FocalLoss         — down-weights easy, well-classified examples so the optimiser
    focuses on the hard/under-represented ones. A strong choice for class imbalance.

Focal loss (Lin et al., 2017), multi-class form:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
where p_t is the softmax probability assigned to the TRUE class. gamma controls how
hard the modulation is (gamma=0 -> plain CE); alpha_t is an optional per-class weight.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weighting."""

    def __init__(self, alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # `alpha` is a (C,) tensor of per-class weights (e.g. inverse frequency),
        # registered as a buffer so it follows the module across .to(device).
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C) raw scores ; target: (B,) class indices
        logp = F.log_softmax(logits, dim=1)                       # (B, C)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)    # (B,) log prob of true class
        pt = logpt.exp()                                          # (B,) prob of true class

        focal = (1.0 - pt) ** self.gamma * (-logpt)               # (B,) modulated CE

        if self.alpha is not None:
            at = self.alpha.gather(0, target)                     # (B,) per-sample class weight
            focal = at * focal

        if self.reduction == "sum":
            return focal.sum()
        if self.reduction == "none":
            return focal
        return focal.mean()


def build_criterion(cfg, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """Return the loss module selected in config.

    cfg.loss_type == "focal"          -> FocalLoss(alpha=class_weights, gamma=cfg.focal_gamma)
    cfg.loss_type == "cross_entropy"  -> nn.CrossEntropyLoss(weight=class_weights, label_smoothing=...)
    """
    if cfg.loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=cfg.focal_gamma)
    if cfg.loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights,
                                   label_smoothing=cfg.label_smoothing)
    raise ValueError(f"Unknown loss_type '{cfg.loss_type}'. "
                     f"Options: 'cross_entropy', 'focal'.")
