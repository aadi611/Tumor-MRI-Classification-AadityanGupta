"""
train.py
========
Command-line orchestrator for the modular pipeline.

Examples
--------
Train a single model:
    python train.py --model resnet50

Train & compare the whole portfolio, then write a leaderboard:
    python train.py --compare

Use the ORIGINAL Kaggle Training/Testing split (keeps the distribution shift):
    python train.py --model densenet121 --no-resplit

The script:
  1. seeds everything, builds dataloaders (default: pooled stratified 70/15/15),
  2. runs progressive fine-tuning (Phase 1 head -> Phase 2 top layers),
  3. evaluates on the held-out test split (classification report, CM, ROC-AUC),
  4. saves the best checkpoint + figures + a JSON leaderboard.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys

import numpy as np

# Force UTF-8 stdout/stderr so Unicode (box-drawing chars in logs, etc.) survives
# Windows' default cp1252 console encoding when output is redirected to a file.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src import (CFG, Config, TemperatureScaling, build_dataloaders, build_model,
                 compute_class_weights, describe_splits, ece_score, ensure_dirs,
                 evaluate_model, plot_history, save_run_config, set_seed,
                 train_model)
from src.config import MODELS_DIR, RESULTS_DIR
from src.evaluate import predict
from src.losses import build_criterion


def run_one(model_name: str, loaders, splits, cfg: Config) -> dict:
    """Train + evaluate a single architecture, return its test metrics."""
    device = cfg.device
    class_weights = compute_class_weights(splits["train"], device) if cfg.use_class_weights else None
    criterion = build_criterion(cfg, class_weights)

    model = build_model(model_name, dropout=cfg.dropout_head)
    ckpt = MODELS_DIR / f"{model_name}_best.pth"

    history = train_model(model, loaders, criterion, cfg, ckpt)
    plot_history(history, model_name)

    # Persist the per-epoch history so pt_06 can overlay training curves.
    with open(RESULTS_DIR / f"{model_name}_history.pkl", "wb") as fh:
        pickle.dump(history, fh)

    metrics = evaluate_model(model, loaders["test"], cfg, tag=model_name)
    metrics["checkpoint"] = str(ckpt)
    metrics["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6

    # Persist raw predictions for pt_06 (McNemar test + calibration analysis).
    # y_true is saved too — the McNemar test needs ground truth to align on.
    np.save(RESULTS_DIR / f"{model_name}_y_true.npy", metrics["y_true"])
    np.save(RESULTS_DIR / f"{model_name}_y_pred.npy", metrics["y_pred"])
    np.save(RESULTS_DIR / f"{model_name}_y_prob.npy", metrics["y_prob"])

    # Post-hoc temperature calibration: fit on val, measure the calibrated ECE
    # on test so pt_06 can show ECE before vs after for every model.
    scaled = TemperatureScaling(model)
    metrics["temperature"] = scaled.fit(loaders["val"], device)
    y_true_c, _, y_prob_c = predict(scaled, loaders["test"], device, use_tta=False)
    metrics["ece_calibrated"] = ece_score(y_true_c, y_prob_c)
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Brain-tumor MRI training pipeline")
    ap.add_argument("--model", default=None, help="single model to train")
    ap.add_argument("--compare", action="store_true", help="train all models in the portfolio")
    ap.add_argument("--no-resplit", action="store_true",
                    help="use the original Kaggle Training/Testing folders")
    ap.add_argument("--epochs1", type=int, default=None, help="override Phase 1 epochs")
    ap.add_argument("--epochs2", type=int, default=None, help="override Phase 2 epochs")
    ap.add_argument("--workers", type=int, default=None, help="DataLoader worker processes")
    ap.add_argument("--loss", choices=["cross_entropy", "focal"], default=None,
                    help="loss function (default from config: cross_entropy)")
    args = ap.parse_args()

    cfg = Config()
    if args.no_resplit:
        cfg.resplit_data = False
    if args.loss is not None:
        cfg.loss_type = args.loss
    if args.epochs1 is not None:
        cfg.phase1_epochs = args.epochs1
    if args.epochs2 is not None:
        cfg.phase2_epochs = args.epochs2
    if args.workers is not None:
        cfg.num_workers = args.workers

    ensure_dirs()
    set_seed(cfg.seed)
    print(f"Device: {cfg.device} | resplit_data={cfg.resplit_data} | AMP={cfg.use_amp}")

    loaders, splits = build_dataloaders(cfg)
    describe_splits(splits)
    save_run_config(cfg, extra={"split_sizes": {k: len(v) for k, v in splits.items()}})

    # Decide which models to run.
    if args.compare:
        names = cfg.model_names
    elif args.model:
        names = [args.model]
    else:
        names = ["resnet50"]  # sensible default single run

    results = []
    for name in names:
        print(f"\n{'='*70}\nTRAINING: {name}\n{'='*70}")
        results.append(run_one(name, loaders, splits, cfg))

    # ── Leaderboard ─────────────────────────────────────────────────────
    results.sort(key=lambda r: r["macro_f1"], reverse=True)
    print(f"\n{'='*70}\nLEADERBOARD (sorted by macro-F1)\n{'='*70}")
    print(f"{'model':16s} {'acc':>7s} {'macroF1':>8s} {'wF1':>7s} {'ROC-AUC':>8s}")
    for r in results:
        print(f"{r['tag']:16s} {r['accuracy']:7.3f} {r['macro_f1']:8.3f} "
              f"{r['weighted_f1']:7.3f} {r['macro_roc_auc']:8.3f}")

    # Drop non-JSON-serialisable / bulky keys: the full report and the raw
    # prediction arrays (the latter are already saved alongside as .npy).
    _SKIP = {"report", "y_true", "y_pred", "y_prob"}
    leaderboard = RESULTS_DIR / "leaderboard.json"
    leaderboard.write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in _SKIP} for r in results], indent=2))
    print(f"\nSaved leaderboard -> {leaderboard}")
    if results:
        print(f"Best model: {results[0]['tag']} "
              f"(macro-F1 {results[0]['macro_f1']:.3f})")


if __name__ == "__main__":
    main()
