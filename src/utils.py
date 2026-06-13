"""
utils.py
========
Small cross-cutting helpers: reproducible seeding and run-config logging.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import RESULTS_DIR, Config


def set_seed(seed: int) -> None:
    """Seed every RNG so a run is reproducible end-to-end."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN trades a little speed for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_run_config(cfg: Config, extra: dict | None = None) -> Path:
    """Persist the exact config (and any extra metadata) used for a run."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = cfg.to_dict()
    if extra:
        payload.update(extra)
    out = RESULTS_DIR / "run_config.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def describe_splits(splits: dict) -> None:
    """Print class counts for each split — quick sanity check on balance."""
    from collections import Counter
    from .config import CLASS_LABELS
    for name, samples in splits.items():
        counts = Counter(lbl for _, lbl in samples)
        dist = {CLASS_LABELS[i]: counts.get(i, 0) for i in range(len(CLASS_LABELS))}
        print(f"{name:5s} | n={len(samples):4d} | {dist}")
