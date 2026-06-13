"""
evaluate.py
===========
Model evaluation + plotting:
  * predict (with optional test-time augmentation)
  * classification_report (precision/recall/F1 per class + macro/weighted)
  * confusion matrix (counts + row-normalised)
  * ROC-AUC, one-vs-rest, per class + macro average, with ROC curves
  * training-curve plots
  * inference_time_benchmark — per-image latency (mean ± std, ms)
  * mcnemar_test — paired significance test between two models' predictions
  * plot_misclassification_deepdive — Grad-CAM on the worst-confused pairs

All figures are written to RESULTS_DIR so they can be dropped straight into the deck.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

from .config import (CLASS_LABELS, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES,
                     RESULTS_DIR, Config)
from .models import ece_score


@torch.no_grad()
def predict(model, loader, device, use_tta: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference. Returns (y_true, y_pred, y_prob).

    TTA: average softmax probabilities of the image and its horizontal flip —
    a cheap, safe boost for symmetric anatomy.
    """
    model.eval()
    ys, preds, probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = F.softmax(logits, dim=1)
        if use_tta:
            p_flip = F.softmax(model(torch.flip(x, dims=[3])), dim=1)
            p = (p + p_flip) / 2.0
        ys.append(y.numpy())
        preds.append(p.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
    return (np.concatenate(ys), np.concatenate(preds), np.concatenate(probs))


def macro_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """One-vs-rest macro ROC-AUC."""
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    return roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")


def evaluate_model(model, loader, cfg: Config, tag: str) -> Dict:
    """Full evaluation. Saves confusion matrix + ROC plots, returns a metrics dict."""
    device = cfg.device
    y_true, y_pred, y_prob = predict(model, loader, device, cfg.use_tta)

    report = classification_report(
        y_true, y_pred, target_names=list(CLASS_LABELS), digits=3, output_dict=True)
    report_txt = classification_report(
        y_true, y_pred, target_names=list(CLASS_LABELS), digits=3)
    auc = macro_roc_auc(y_true, y_prob)

    print(f"\n──── {tag} ────")
    print(report_txt)
    print(f"Macro ROC-AUC : {auc:.4f}")
    print(f"Accuracy      : {report['accuracy']:.4f}")
    print(f"Macro F1      : {report['macro avg']['f1-score']:.4f}")

    _plot_confusion(y_true, y_pred, tag)
    _plot_roc(y_true, y_prob, tag)

    # Calibration (raw, uncalibrated) + latency, so the leaderboard carries the
    # full comparison-dashboard column set.
    ece = ece_score(y_true, y_prob)
    mean_ms, std_ms = inference_time_benchmark(model, loader, device)

    # Raw prediction arrays are returned (not written here) so the caller decides
    # whether to persist them — train.py saves them as .npy for pt_06 to reload.
    return {
        "tag": tag,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_roc_auc": auc,
        "ece": ece,
        "inference_ms": mean_ms,
        "inference_ms_std": std_ms,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "report": report,
    }


# ──────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────
def _plot_confusion(y_true, y_pred, tag: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_LABELS).plot(
        ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
    axes[0].set_title(f"{tag} — counts")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=axes[1])
    axes[1].set_title(f"{tag} — row-normalised recall")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    plt.tight_layout()
    out = RESULTS_DIR / f"{tag}_confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.name}")


def _plot_roc(y_true, y_prob, tag: str) -> None:
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, name in enumerate(CLASS_LABELS):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_bin[:, i], y_prob[:, i]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{tag} — ROC (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out = RESULTS_DIR / f"{tag}_roc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.name}")


def plot_history(history, tag: str) -> None:
    """Plot loss + accuracy curves over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(history.train_loss, label="train")
    axes[0].plot(history.val_loss, label="val")
    axes[0].set_title(f"{tag} — loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history.train_acc, label="train")
    axes[1].plot(history.val_acc, label="val")
    axes[1].set_title(f"{tag} — accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout()
    out = RESULTS_DIR / f"{tag}_training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.name}")


# ──────────────────────────────────────────────────────────────────────────
# Inference latency benchmark
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def inference_time_benchmark(model, loader, device, n_batches: int = 20
                             ) -> Tuple[float, float]:
    """Wall-clock inference latency per image. Returns (mean_ms, std_ms).

    Warms up with 3 untimed batches (CUDA kernel compilation, cache effects),
    then times `n_batches` forward passes. On CUDA we synchronise around each
    pass — without it, timings only measure async kernel *launch*, not compute.
    The std is across batch means, so it reflects run-to-run jitter.
    """
    model.eval()
    model.to(device)
    is_cuda = device.type == "cuda"

    def batches(n):
        """Yield up to n batches, re-iterating the loader if it is shorter."""
        served = 0
        while served < n:
            for x, _ in loader:
                if served == n:
                    return
                yield x
                served += 1

    for x in batches(3):                          # warm-up, untimed
        model(x.to(device, non_blocking=True))

    per_image_ms = []
    for x in batches(n_batches):
        x = x.to(device, non_blocking=True)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if is_cuda:
            torch.cuda.synchronize()
        per_image_ms.append((time.perf_counter() - t0) / x.size(0) * 1000.0)

    mean_ms = float(np.mean(per_image_ms))
    std_ms = float(np.std(per_image_ms))
    print(f"Inference latency: {mean_ms:.2f} ± {std_ms:.2f} ms/image "
          f"({device.type}, {len(per_image_ms)} batches)")
    return mean_ms, std_ms


# ──────────────────────────────────────────────────────────────────────────
# McNemar's test — paired model comparison
# ──────────────────────────────────────────────────────────────────────────
def mcnemar_test(y_pred_a, y_pred_b, y_true) -> Tuple[float, float]:
    """McNemar's test on two models' predictions over the SAME test samples.

    Builds the 2x2 contingency table of per-sample correctness:

                          B correct   B wrong
        A correct            n11        n10
        A wrong              n01        n00

    Only the discordant cells matter — n10 (A right, B wrong) and n01 (A wrong,
    B right). Under H0 (equal error rates) they are equally likely, giving the
    continuity-corrected statistic  chi2 = (|n10 - n01| - 1)^2 / (n10 + n01),
    which is chi-square with 1 degree of freedom.

    Returns (chi2_statistic, p_value).
    """
    y_pred_a, y_pred_b, y_true = map(np.asarray, (y_pred_a, y_pred_b, y_true))
    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true

    n11 = int(np.sum(a_correct & b_correct))
    n10 = int(np.sum(a_correct & ~b_correct))
    n01 = int(np.sum(~a_correct & b_correct))
    n00 = int(np.sum(~a_correct & ~b_correct))

    print(f"Contingency table:  both correct={n11}  A-only={n10}  "
          f"B-only={n01}  both wrong={n00}")

    if n10 + n01 == 0:        # models disagree on nothing — no evidence either way
        stat, p_value = 0.0, 1.0
    else:
        stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
        p_value = float(chi2_dist.sf(stat, df=1))

    verdict = ("SIGNIFICANT difference between the models (p < 0.05)"
               if p_value < 0.05 else
               "no significant difference between the models (p >= 0.05)")
    print(f"McNemar chi2 = {stat:.4f}, p = {p_value:.4f} -> {verdict}")
    return float(stat), p_value


# ──────────────────────────────────────────────────────────────────────────
# Misclassification deep-dive with Grad-CAM
# ──────────────────────────────────────────────────────────────────────────
def _denormalize_to_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """(3,H,W) ImageNet-normalised tensor -> (H,W,3) uint8 RGB for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (img_tensor.cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def plot_misclassification_deepdive(model, loader, device, gradcam,
                                    top_n: int = 3, tag: str = "model") -> None:
    """Grad-CAM deep-dive into the model's worst-confused class pairs.

    1. Predicts over `loader` and ranks off-diagonal confusion-matrix cells.
    2. For the `top_n` most confused (true -> predicted) pairs, takes up to 3
       misclassified examples each and renders their Grad-CAM overlays —
       showing WHERE the model looked when it got these wrong.

    `gradcam` is a GradCAM instance (see pt_04) already hooked to `model`'s
    target layer; its .generate() needs gradients, so no torch.no_grad() here.
    Saves to results/{tag}_misclassification_deepdive.png.
    """
    model.eval()

    # Pass over the loader: predictions + the misclassified images themselves.
    y_true, y_pred, wrong_imgs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(1).cpu()
            y_true.append(y.numpy()); y_pred.append(preds.numpy())
            for img, t, p in zip(x, y, preds):
                if t != p:
                    wrong_imgs.append((img, int(t), int(p)))
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)

    # Rank off-diagonal cells of the confusion matrix.
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    np.fill_diagonal(cm, 0)
    flat = [(cm[i, j], i, j) for i in range(NUM_CLASSES)
            for j in range(NUM_CLASSES) if cm[i, j] > 0]
    flat.sort(reverse=True)
    pairs = [(i, j) for _, i, j in flat[:top_n]]
    if not pairs:
        print(f"[{tag}] no misclassifications — nothing to deep-dive into")
        return

    n_cols = 3
    fig, axes = plt.subplots(len(pairs), n_cols,
                             figsize=(4 * n_cols, 4.2 * len(pairs)), squeeze=False)
    for row, (t_cls, p_cls) in enumerate(pairs):
        examples = [w for w in wrong_imgs if w[1] == t_cls and w[2] == p_cls][:n_cols]
        for col in range(n_cols):
            ax = axes[row][col]
            ax.axis("off")
            if col >= len(examples):
                continue
            img, _, _ = examples[col]
            heatmap, pred_idx, conf = gradcam.generate(
                img.unsqueeze(0).to(device))
            rgb = _denormalize_to_rgb(img)
            hm = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]))
            hm_color = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
            hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
            overlay = np.uint8(rgb * 0.55 + hm_color * 0.45)
            ax.imshow(overlay)
            ax.set_title(f"True: {CLASS_LABELS[t_cls]}\n"
                         f"Pred: {CLASS_LABELS[pred_idx]} ({conf:.2f})",
                         fontsize=10)
        axes[row][0].set_ylabel(f"{CLASS_LABELS[t_cls]} -> {CLASS_LABELS[p_cls]}",
                                fontsize=11)

    fig.suptitle(f"{tag} — top-{len(pairs)} confused pairs (Grad-CAM)", y=1.0)
    plt.tight_layout()
    out = RESULTS_DIR / f"{tag}_misclassification_deepdive.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.name}")
