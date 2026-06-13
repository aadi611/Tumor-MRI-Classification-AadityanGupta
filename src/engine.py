"""
engine.py
=========
The training engine: epoch loops, AMP, early stopping, checkpointing,
ReduceLROnPlateau, and the two-phase progressive fine-tuning orchestrator.

Every training best-practice the brief asked for lives here:
  * AMP (mixed precision) for speed/memory on CUDA
  * Early stopping on validation loss (with min-delta + patience)
  * Best-checkpoint saving (lowest val loss wins)
  * LR scheduling: ReduceLROnPlateau (default) or CosineAnnealingWarmRestarts
  * Optional MixUp augmentation on the training pass (cfg.use_mixup)
  * Gradient-aware optimiser (only trainable params are passed in);
    Phase 2 uses differential LRs (head 10x the backbone)
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import NUM_CLASSES, Config
from .losses import FocalLoss
from .models import _head_param_ids, count_trainable, freeze_backbone, unfreeze_top


# ──────────────────────────────────────────────────────────────────────────
# Early stopping
# ──────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """Stop training when val loss stops improving by `min_delta` for `patience`."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Returns True if this is a new best (so the caller can checkpoint)."""
        improved = val_loss < self.best - self.min_delta
        if improved:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


@dataclass
class History:
    """Per-epoch metric trace, used for plotting training curves."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────
# Scheduler factory
# ──────────────────────────────────────────────────────────────────────────
def _build_scheduler(optimizer, cfg: Config):
    """Build the LR scheduler selected by cfg.lr_scheduler.

    "plateau" -> ReduceLROnPlateau, stepped once per epoch on val loss.
    "cosine"  -> CosineAnnealingWarmRestarts, stepped per BATCH with a
                 fractional epoch index (see _run_epoch).
    """
    if cfg.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2)
    if cfg.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.lr_plateau_factor,
            patience=cfg.lr_plateau_patience)
    raise ValueError(f"Unknown lr_scheduler '{cfg.lr_scheduler}'. "
                     f"Options: 'plateau', 'cosine'.")


# ──────────────────────────────────────────────────────────────────────────
# Single-epoch passes
# ──────────────────────────────────────────────────────────────────────────
def _mixup_batch(x, y, alpha: float, num_classes: int):
    """MixUp (Zhang et al., 2018): convex-combine a batch with a shuffled
    copy of itself. Returns mixed inputs and SOFT label targets."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1.0 - lam) * x[perm]
    y_onehot = F.one_hot(y, num_classes).float()
    y_soft = lam * y_onehot + (1.0 - lam) * y_onehot[perm]
    return x_mixed, y_soft


def _run_epoch(model, loader, criterion, device, optimizer=None,
               scaler=None, use_amp=False, scheduler=None, epoch_idx=0,
               use_mixup=False, mixup_alpha=0.2):
    """One pass over `loader`. Train mode if optimizer is given, else eval.

    `scheduler` is only passed for per-batch schedulers (cosine warm restarts),
    which are stepped with a fractional epoch index after each optimizer step.
    `use_mixup` applies only to the training pass; soft targets go through
    F.cross_entropy (the criterion object may not support them, e.g. FocalLoss).
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_i, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            mixed = is_train and use_mixup
            if mixed:
                x_in, y_soft = _mixup_batch(x, y, mixup_alpha, NUM_CLASSES)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x_in if mixed else x)
                if mixed:
                    # Soft targets: criterion objects (FocalLoss, weighted CE)
                    # expect hard labels, so use functional CE; carry over the
                    # class weights if the criterion has them.
                    loss = F.cross_entropy(logits, y_soft,
                                           weight=getattr(criterion, "weight", None))
                else:
                    loss = criterion(logits, y)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:   # per-batch cosine warm restarts
                    scheduler.step(epoch_idx + batch_i / n_batches)

            total_loss += loss.item() * x.size(0)
            # Under MixUp, accuracy vs the dominant original label is an
            # approximation — exact train acc is not defined for mixed targets.
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────────────────
# Phase trainer
# ──────────────────────────────────────────────────────────────────────────
def _train_phase(model, loaders, criterion, optimizer, scheduler, cfg: Config,
                 epochs: int, ckpt_path: Path, history: History, phase_name: str):
    device = cfg.device
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
    stopper = EarlyStopping(cfg.early_stop_patience, cfg.early_stop_min_delta)
    best_state = copy.deepcopy(model.state_dict())

    # Cosine warm restarts step per BATCH (inside _run_epoch); plateau per epoch.
    per_batch_sched = isinstance(
        scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)

    # MixUp needs soft targets -> functional CE. FocalLoss has no soft-target
    # form, so MixUp is silently incompatible — skip it and say so once.
    use_mixup = cfg.use_mixup and not isinstance(criterion, FocalLoss)
    if cfg.use_mixup and not use_mixup:
        print(f"[{phase_name}] use_mixup=True ignored: FocalLoss does not "
              f"support soft targets")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, loaders["train"], criterion, device,
                                     optimizer, scaler, cfg.use_amp,
                                     scheduler=scheduler if per_batch_sched else None,
                                     epoch_idx=epoch - 1,
                                     use_mixup=use_mixup,
                                     mixup_alpha=cfg.mixup_alpha)
        va_loss, va_acc = _run_epoch(model, loaders["val"], criterion, device,
                                     use_amp=cfg.use_amp)
        if not per_batch_sched:
            scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        history.train_loss.append(tr_loss); history.val_loss.append(va_loss)
        history.train_acc.append(tr_acc);   history.val_acc.append(va_acc)
        history.lr.append(lr_now)

        is_best = stopper.step(va_loss)
        if is_best:
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)

        print(f"[{phase_name}] epoch {epoch:02d}/{epochs} "
              f"| train {tr_loss:.4f}/{tr_acc:.3f} "
              f"| val {va_loss:.4f}/{va_acc:.3f} "
              f"| lr {lr_now:.2e} | {time.time()-t0:.1f}s"
              f"{'  <-- best' if is_best else ''}")

        if stopper.should_stop:
            print(f"[{phase_name}] early stopping at epoch {epoch} "
                  f"(best val loss {stopper.best:.4f})")
            break

    model.load_state_dict(best_state)   # restore best weights
    return model


def _train_from_scratch(model, loaders: Dict[str, DataLoader], criterion,
                        cfg: Config, ckpt_path: Path) -> History:
    """Single-phase, full-network training for from-scratch models.

    There is no pretrained backbone to freeze, so the two-phase head-then-
    fine-tune schedule does not apply. We train every parameter from random
    init at a moderate LR for `cfg.scratch_epochs` (with the same early-stop,
    checkpoint and ReduceLROnPlateau machinery as the transfer-learning path).
    """
    device = cfg.device
    model.to(device)
    history = History()

    for p in model.parameters():        # everything trainable from the start
        p.requires_grad = True

    print(f"\n=== From-scratch training ({model.arch_name}) | trainable params: "
          f"{count_trainable(model):,} ===")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.scratch_lr,
                            weight_decay=cfg.weight_decay)
    sch = _build_scheduler(opt, cfg)
    _train_phase(model, loaders, criterion, opt, sch, cfg,
                 cfg.scratch_epochs, ckpt_path, history, "SCRATCH")
    return history


def train_model(model, loaders: Dict[str, DataLoader], criterion,
                cfg: Config, ckpt_path: Path) -> History:
    """Train a model and return its History.

    * From-scratch models (e.g. our CustomBrainCNN): one full-network phase.
    * Pretrained backbones: two-phase progressive fine-tuning — Phase 1 trains
      the new head with the backbone frozen, Phase 2 unfreezes the top layers.
    """
    # Route from-scratch architectures away from the freeze/unfreeze schedule.
    if getattr(model, "from_scratch", False):
        return _train_from_scratch(model, loaders, criterion, cfg, ckpt_path)

    device = cfg.device
    model.to(device)
    history = History()

    # ── Phase 1: frozen backbone, train classifier head only ───────────
    freeze_backbone(model)
    print(f"\n=== Phase 1 ({model.arch_name}) | trainable params: "
          f"{count_trainable(model):,} ===")
    opt1 = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.phase1_lr, weight_decay=cfg.weight_decay,
    )
    sch1 = _build_scheduler(opt1, cfg)
    model = _train_phase(model, loaders, criterion, opt1, sch1, cfg,
                         cfg.phase1_epochs, ckpt_path, history, "P1")

    # ── Phase 2: unfreeze top fraction, fine-tune at very low LR ────────
    unfreeze_top(model, cfg.unfreeze_fraction)
    print(f"\n=== Phase 2 ({model.arch_name}) | trainable params: "
          f"{count_trainable(model):,} ===")
    # Differential LRs: the freshly-initialised head can absorb a 10x larger
    # step than the pretrained backbone weights it sits on.
    head_ids = _head_param_ids(model)
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in head_ids]
    head_params = [p for p in model.parameters()
                   if p.requires_grad and id(p) in head_ids]
    opt2 = torch.optim.AdamW(
        [{"params": backbone_params, "lr": cfg.phase2_lr},
         {"params": head_params, "lr": cfg.phase2_lr * 10}],
        weight_decay=cfg.weight_decay,
    )
    sch2 = _build_scheduler(opt2, cfg)
    model = _train_phase(model, loaders, criterion, opt2, sch2, cfg,
                         cfg.phase2_epochs, ckpt_path, history, "P2")

    return history
