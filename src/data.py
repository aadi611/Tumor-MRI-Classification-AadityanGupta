"""
data.py
=======
Datasets, transforms (augmentation pipeline), stratified splitting and dataloaders.

Highlights
----------
* `build_transforms` — separate train (augmented) and eval (deterministic) pipelines,
  both prefixed with the domain-specific MRIPreprocess (crop + CLAHE).
* `MRIDataset` — a lightweight (path, label) dataset so the SAME sample list can be
  given different transforms (train vs eval) without the Subset transform-sharing bug.
* `build_dataloaders` — two modes:
    1. resplit_data=True  -> pool Training+Testing, stratified 70/15/15 (fixes the
       distribution-shift artefact in the Kaggle dataset).
    2. resplit_data=False -> use the original folders, stratified val split from train.
* `compute_class_weights` — inverse-frequency weights for the loss function.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import (CLASSES, IMAGENET_MEAN, IMAGENET_STD, TEST_DIR, TRAIN_DIR,
                     Config)
from .preprocessing import MRIPreprocess

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ──────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────
def build_transforms(cfg: Config) -> Tuple[Callable, Callable]:
    """Return (train_transform, eval_transform).

    Train: domain preprocess -> medical-safe augmentation -> normalise.
    Eval : domain preprocess -> deterministic resize -> normalise.
    """
    pre = MRIPreprocess(
        use_crop=cfg.use_crop, use_clahe=cfg.use_clahe,
        clahe_clip_limit=cfg.clahe_clip_limit, clahe_grid=cfg.clahe_grid,
    )
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    train_tf = transforms.Compose([
        pre,
        transforms.RandomResizedCrop(cfg.img_size, scale=cfg.aug_zoom),
        transforms.RandomHorizontalFlip(p=cfg.aug_hflip_p),
        transforms.RandomRotation(cfg.aug_rotation_deg),
        transforms.RandomAffine(degrees=0, translate=(cfg.aug_translate, cfg.aug_translate)),
        transforms.ColorJitter(brightness=cfg.aug_brightness, contrast=cfg.aug_contrast),
        transforms.RandomGrayscale(p=0.05),   # rare channel-drop; MRI is near-grayscale anyway
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.0), value=0),
    ])

    eval_tf = transforms.Compose([
        pre,
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


# ──────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────
class MRIDataset(Dataset):
    """Dataset over an explicit list of (path, label) pairs.

    Decoupling the file list from the transform lets us apply train-augmentation
    to the training indices and deterministic eval-transforms to val/test indices
    while they all originate from the same pooled sample list.
    """

    def __init__(self, samples: List[Tuple[Path, int]], transform: Callable):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def _scan_dir(root: Path) -> List[Tuple[Path, int]]:
    """Collect (path, class_index) for every image under root/<class>/."""
    samples: List[Tuple[Path, int]] = []
    for ci, cls in enumerate(CLASSES):
        cdir = root / cls
        if not cdir.is_dir():
            continue
        for p in cdir.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                samples.append((p, ci))
    return samples


# ──────────────────────────────────────────────────────────────────────────
# Splitting + dataloaders
# ──────────────────────────────────────────────────────────────────────────
def _stratified_split(samples, labels, test_size, seed):
    idx = np.arange(len(samples))
    a_idx, b_idx = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=seed
    )
    return [samples[i] for i in a_idx], [samples[i] for i in b_idx]


def build_dataloaders(cfg: Config):
    """Build train/val/test dataloaders.

    Returns
    -------
    loaders : dict with keys 'train', 'val', 'test'
    splits  : dict of raw sample lists (for inspection / logging)
    """
    train_tf, eval_tf = build_transforms(cfg)

    if cfg.resplit_data:
        # Pool everything, then stratified 70/15/15 (defaults).
        pool = _scan_dir(TRAIN_DIR) + _scan_dir(TEST_DIR)
        labels = [lbl for _, lbl in pool]
        # First peel off the test set, then split the remainder into train/val.
        train_val, test = _stratified_split(pool, labels, cfg.test_split, cfg.seed)
        tv_labels = [lbl for _, lbl in train_val]
        # val fraction is relative to the train_val remainder.
        val_rel = cfg.val_split / (1.0 - cfg.test_split)
        train, val = _stratified_split(train_val, tv_labels, val_rel, cfg.seed)
    else:
        # Use the dataset's own Training/Testing split; carve val out of train.
        train_pool = _scan_dir(TRAIN_DIR)
        labels = [lbl for _, lbl in train_pool]
        train, val = _stratified_split(train_pool, labels, cfg.val_split, cfg.seed)
        test = _scan_dir(TEST_DIR)

    splits = {"train": train, "val": val, "test": test}
    datasets = {
        "train": MRIDataset(train, train_tf),
        "val": MRIDataset(val, eval_tf),
        "test": MRIDataset(test, eval_tf),
    }
    loaders = {
        name: DataLoader(
            ds, batch_size=cfg.batch_size,
            shuffle=(name == "train"),
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        for name, ds in datasets.items()
    }
    return loaders, splits


def compute_class_weights(train_samples, device) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss (counters imbalance)."""
    labels = np.array([lbl for _, lbl in train_samples])
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero
    weights = counts.sum() / (len(CLASSES) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)
