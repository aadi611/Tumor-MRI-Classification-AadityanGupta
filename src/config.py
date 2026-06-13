"""
config.py
=========
Single source of truth for every tunable knob in the pipeline.

Keeping configuration in one dataclass (instead of scattering magic numbers across
notebooks) is a production best-practice: experiments become reproducible, a run can
be logged/serialised in full, and swapping hyper-parameters never requires touching
model or training code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple

import torch

# ──────────────────────────────────────────────────────────────────────────
# Paths — anchored to the project root so the code is location-independent.
# ──────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# The four diagnostic classes (folder names == class names after the rename fix).
CLASSES: Tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")
CLASS_LABELS: Tuple[str, ...] = ("Glioma", "Meningioma", "No Tumor", "Pituitary")
NUM_CLASSES = len(CLASSES)

# ImageNet normalisation stats — required because we use ImageNet-pretrained backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Config:
    """All experiment hyper-parameters in one serialisable object."""

    # ── Reproducibility ────────────────────────────────────────────────
    seed: int = 42

    # ── Data ────────────────────────────────────────────────────────────
    img_size: int = 224                 # input resolution fed to the network
    batch_size: int = 32
    num_workers: int = 0                # 0 is safest on Windows (no fork)
    val_split: float = 0.15             # fraction of TRAIN used for validation
    test_split: float = 0.15            # used only when `resplit_data=True`

    # The single most important switch in this project:
    #   The Kaggle (Sartaj) Training/Testing folders have a documented
    #   distribution shift -> models score ~88% on val but ~65% on the
    #   supplied Test set. Pooling all images and re-splitting stratified
    #   removes that artefact and lets us measure true generalisation.
    resplit_data: bool = True

    # ── Preprocessing toggles ──────────────────────────────────────────
    use_crop: bool = True               # crop to the brain bounding box
    use_clahe: bool = True              # contrast-limited adaptive hist. equalisation
    clahe_clip_limit: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)

    # ── Augmentation (medical-imaging safe) ────────────────────────────
    aug_rotation_deg: int = 15          # small only — anatomy must stay plausible
    aug_translate: float = 0.08
    aug_zoom: Tuple[float, float] = (0.9, 1.0)
    aug_brightness: float = 0.10
    aug_contrast: float = 0.10
    aug_hflip_p: float = 0.5            # left/right brains are both valid
    # NOTE: we deliberately do NOT use vertical flips or heavy distortion —
    # they create anatomically impossible scans and hurt a medical model.

    # ── Optimisation ───────────────────────────────────────────────────
    label_smoothing: float = 0.0        # dropped vs the old run (hurt minority recall)
    weight_decay: float = 1e-4
    use_class_weights: bool = True      # counter class imbalance in the loss
    use_amp: bool = field(default_factory=lambda: torch.cuda.is_available())

    # ── Loss function ──────────────────────────────────────────────────
    # "cross_entropy" (proper, calibrated default) or "focal" (hard-example
    # mining for class imbalance). Resolved by losses.build_criterion().
    loss_type: str = "cross_entropy"
    focal_gamma: float = 2.0            # focusing parameter (only used for focal loss)

    # ── LR scheduler ───────────────────────────────────────────────────
    # "plateau" -> ReduceLROnPlateau (default, steps per epoch on val loss)
    # "cosine"  -> CosineAnnealingWarmRestarts(T_0=10, T_mult=2), steps per batch
    lr_scheduler: str = "plateau"

    # ── MixUp augmentation (training pass only) ────────────────────────
    # Mixes pairs of samples + their one-hot labels with lambda ~ Beta(a, a).
    # Ignored when loss_type == "focal" (FocalLoss has no soft-target form).
    use_mixup: bool = False
    mixup_alpha: float = 0.2

    # ── From-scratch training (CustomBrainCNN) ─────────────────────────
    # No pretrained backbone -> a single full-network phase instead of the
    # two-phase freeze/unfreeze schedule, at a moderate LR.
    scratch_epochs: int = 40
    scratch_lr: float = 3e-4

    # ── Progressive fine-tuning schedule ───────────────────────────────
    # Phase 1: backbone frozen, train only the new classifier head.
    phase1_epochs: int = 12
    phase1_lr: float = 1e-3
    # Phase 2: unfreeze the top ~30% of backbone layers, very low LR.
    phase2_epochs: int = 25
    phase2_lr: float = 1e-5
    unfreeze_fraction: float = 0.30     # fraction of backbone params to unfreeze

    # ── Training-loop best practices ───────────────────────────────────
    early_stop_patience: int = 7        # epochs without val-loss improvement
    early_stop_min_delta: float = 1e-4
    lr_plateau_patience: int = 3        # ReduceLROnPlateau patience
    lr_plateau_factor: float = 0.3
    dropout_head: float = 0.4

    # ── Model portfolio ────────────────────────────────────────────────
    # Names resolved by models.build_model(). Train one or compare all.
    # vgg16 dropped from the default portfolio (slow, outclassed) in favour of
    # efficientnet_b3; it remains available via `--model vgg16`. mobilenet_v3 is
    # the lightweight baseline in the comparison.
    model_names: List[str] = field(
        default_factory=lambda: ["resnet50", "densenet121", "efficientnet_b3",
                                 "efficientnet_b0", "mobilenet_v3"]
    )

    # ── Evaluation ──────────────────────────────────────────────────────
    use_tta: bool = True                # test-time augmentation (h-flip avg)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot (for logging the exact run config)."""
        d = asdict(self)
        d["device"] = str(self.device)
        return d


# A module-level default instance for convenience / quick imports.
CFG = Config()


def ensure_dirs() -> None:
    """Create output directories if they don't yet exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
