# Architecture & Design Reference

This document provides an in-depth walkthrough of every architectural and engineering decision in the project — from raw pixel to final prediction. It is intended as a standalone reference for understanding the pipeline, reproducing experiments, or extending the codebase.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Pipeline](#2-data-pipeline)
3. [MRI Preprocessing](#3-mri-preprocessing)
4. [Model Portfolio](#4-model-portfolio)
   - 4.1 [Shared Classifier Head](#41-shared-classifier-head)
   - 4.2 [Transfer Learning Models](#42-transfer-learning-models)
   - 4.3 [Custom CNN (From Scratch)](#43-custom-cnn-from-scratch)
   - 4.4 [Classical Baseline — HOG + SVM](#44-classical-baseline--hog--svm)
5. [Training Pipeline](#5-training-pipeline)
   - 5.1 [Two-Phase Progressive Fine-Tuning](#51-two-phase-progressive-fine-tuning)
   - 5.2 [From-Scratch Training Schedule](#52-from-scratch-training-schedule)
   - 5.3 [Loss Functions](#53-loss-functions)
   - 5.4 [Optimisation & Learning Rate Scheduling](#54-optimisation--learning-rate-scheduling)
   - 5.5 [Regularisation](#55-regularisation)
   - 5.6 [Automatic Mixed Precision (AMP)](#56-automatic-mixed-precision-amp)
6. [Evaluation Suite](#6-evaluation-suite)
7. [Explainability — Grad-CAM](#7-explainability--grad-cam)
8. [Post-Hoc Calibration](#8-post-hoc-calibration)
9. [Statistical Significance Testing](#9-statistical-significance-testing)
10. [Live Demo Application](#10-live-demo-application)
11. [Configuration Reference](#11-configuration-reference)

---

## 1. System Overview

```
Raw MRI image (JPEG/PNG)
        │
        ▼
┌──────────────────────┐
│   MRIPreprocess      │  Otsu brain-crop → CLAHE contrast enhancement
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   Augmentation       │  Random crop/flip/rotate/jitter  (train only)
│   or Resize          │  Deterministic 224×224           (val/test)
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   ToTensor +         │  [0,255] uint8 → [0,1] float → ImageNet normalise
│   Normalise          │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   Model              │  CustomCNN / ResNet-50 / EfficientNet / etc.
│   (forward pass)     │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   Logits (B × 4)     │  Raw scores — softmax applied at eval only
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   Loss               │  Weighted CrossEntropy or FocalLoss
└──────────────────────┘
        │
   (training loop)
        │
        ▼
┌──────────────────────┐
│   Evaluation         │  Accuracy / Macro-F1 / ROC-AUC / ECE / Latency
└──────────────────────┘
```

**Classes:** `glioma · meningioma · notumor · pituitary`
**Input resolution:** 224 × 224 × 3 (RGB)
**Loss:** class-weighted CrossEntropyLoss (default) or FocalLoss

---

## 2. Data Pipeline

**Source:** `src/data.py`

### Distribution shift — why we re-split

The Kaggle dataset's official `Training/` and `Testing/` folders have a well-documented distribution shift: the glioma test images come from a different scanner/sequence population than the training set. This causes EfficientNet-B0's glioma recall to collapse from ~0.86 (Protocol B) to 0.16 (Protocol A) — making the SVM appear to win, when in reality it just happens to transfer better under this specific shift.

The `src/` pipeline defaults to `resplit_data=True`, which pools all 3,658 images and performs a **stratified 70 / 15 / 15 random split** (seed = 42). This is the honest evaluation protocol.

### Stratified splitting

```
All images (3,658)
        │
        ├── 15% → test   (stratified, seed=42)
        │
        └── 85% → train_val
                    │
                    ├── ~82% → train  (≈ 70% of total)
                    └── ~18% → val    (≈ 15% of total)
```

`val_rel = val_split / (1 - test_split)` ensures the fractions are exact relative to the full pool.

### MRIDataset

A lightweight `(path, label)` dataset, decoupled from transforms. The same sample list is passed to three separate `MRIDataset` instances with different transforms (train augmentation / eval deterministic / eval deterministic), avoiding the `Subset` transform-sharing bug.

### Class weights

Inverse-frequency weights, computed on the training split only:

```
w_c = N_total / (C × N_c)
```

where `N_total` is total training samples, `C` is the number of classes, and `N_c` is the count for class `c`. Passed directly to the loss function to counteract the `notumor` under-representation.

---

## 3. MRI Preprocessing

**Source:** `src/preprocessing.py`

Both steps run as a `PIL → PIL` callable at the front of every transform pipeline, before tensor conversion. They apply identically to training and evaluation (no augmentation in preprocessing).

### Step 1 — Brain-Region Crop

```
RGB MRI
  → Grayscale → GaussianBlur(5×5)
  → Otsu threshold (adaptive to each image's histogram)
  → Morphological: erode × 2, dilate × 2  (removes specks)
  → Find external contours → select largest by area
  → Bounding box → crop (with optional padding)
```

**Why:** Raw MRI slices have a large black border. Cropping means the network's 224×224 input window is entirely occupied by brain tissue, not wasted on uninformative background.

**Robustness:** Falls back to the original image if no contour is detected (near-empty slices, edge cases).

### Step 2 — CLAHE

Contrast-Limited Adaptive Histogram Equalisation, applied to the **L channel** of LAB colour space (not the full RGB). This normalises local contrast per tile without affecting hue/saturation:

```
RGB → LAB
L channel → CLAHE(clipLimit=2.0, tileGrid=8×8)
LAB → RGB
```

**Why CLAHE over global HE:** Global equalisation amplifies noise in low-signal regions. The clip limit caps per-tile amplification to `clipLimit × (tile_area / 256)` histogram bins, keeping noise controlled.

**Why L channel only:** Equalising all three channels independently distorts colour relationships. The L channel carries spatial texture/contrast; equalising it alone preserves colour fidelity.

---

## 4. Model Portfolio

**Source:** `src/models.py`, `src/custom_cnn.py`

All models are accessed through a single factory:

```python
model = build_model("resnet50", dropout=0.4)
```

### 4.1 Shared Classifier Head

Every transfer-learning model has its original final layer replaced by a shared head:

```
in_features
    → Dropout(p=0.4)
    → Linear(in_features → 256)
    → BatchNorm1d(256)
    → ReLU
    → Dropout(p=0.2)
    → Linear(256 → 4)
```

**Design rationale:**
- `BatchNorm1d` on the bottleneck stabilises the gradient signal into the pretrained backbone during Phase 1 (when only the head is trained).
- Two Dropout stages regularise the small medical dataset (3,658 images is modest by ImageNet standards).
- The 256-dim bottleneck adds one non-linear representational step before the final logits.

### 4.2 Transfer Learning Models

| Model | Backbone params | Head in-features | ImageNet weights |
|---|---|---|---|
| ResNet-50 | 23.5 M | 2,048 | IMAGENET1K_V2 |
| EfficientNet-B0 | 4.0 M | 1,280 | IMAGENET1K_V1 |
| EfficientNet-B3 | 10.7 M | 1,536 | IMAGENET1K_V1 |
| DenseNet-121 | 6.9 M | 1,024 | IMAGENET1K_V1 |
| VGG-16 | 134.3 M | 25,088 | IMAGENET1K_V1 |
| MobileNet-V3-Large | 4.2 M | 1,280 | IMAGENET1K_V2 |

**Why these models?**
- **EfficientNet-B0** — best accuracy/parameter trade-off at 4.4 M total params.
- **ResNet-50** — the canonical transfer learning baseline; well-studied, strong.
- **DenseNet-121** — dense connections improve gradient flow; originally designed for medical imaging (CheXNet).
- **MobileNet-V3** — the lightweight baseline; tests whether a 5 M param model can compete.
- **VGG-16** — included for historical completeness (established to be outclassed at 135 M params).

### 4.3 Custom CNN (From Scratch)

**Source:** `src/custom_cnn.py`

The `CustomBrainCNN` is a purpose-built architecture with **no pretrained weights**. It serves as a controlled ablation: by comparing it to ImageNet-pretrained models of similar size, we can measure how much of the transfer models' accuracy comes from pretraining vs. architecture.

#### Architecture

```
Input: (B, 3, 224, 224)

Stem
  Conv2d(3→64, k=7, stride=2, pad=3) → BN → SiLU     (B, 64, 112, 112)
  MaxPool2d(k=3, stride=2, pad=1)                       (B, 64,  56,  56)

Stage 1  [2 × ResidualSEBlock(64→64,  stride=1)]        (B,  64, 56, 56)
Stage 2  [2 × ResidualSEBlock(64→128, stride=2→1)]      (B, 128, 28, 28)
Stage 3  [2 × ResidualSEBlock(128→256, stride=2→1)]     (B, 256, 14, 14)
Stage 4  [2 × ResidualSEBlock(256→512, stride=2→1)]     (B, 512,  7,  7)

Global Average Pooling → (B, 512)
Dropout(0.4)
Linear(512 → 4)  →  raw logits
```

**Total parameters:** ~11.3 M (comparable to ResNet-18)

#### ResidualSEBlock

Each block applies:

```
Main path:
  Conv2d(3×3, stride=s) → BN → SiLU
  Conv2d(3×3, stride=1) → BN
  SEBlock(channel attention)
  Dropout2d (light spatial regularisation, p=0.1)

Skip path:
  Identity           (if in_ch == out_ch and stride == 1)
  Conv2d(1×1) → BN  (otherwise — projection shortcut)

Output: SiLU(main + skip)
```

#### SEBlock (Squeeze-and-Excitation)

```
Input: (B, C, H, W)
  → AdaptiveAvgPool2d(1)    squeeze: (B, C, 1, 1)
  → view(B, C)
  → Linear(C → C/16) → ReLU → Linear(C/16 → C) → Sigmoid
  → view(B, C, 1, 1)
  → x * gate              channel-wise re-weighting
```

The ratio `C/16` is the standard SE bottleneck. The `max(C/16, 4)` guard prevents degenerate bottlenecks in early layers with few channels.

#### Design Choices

| Choice | Rationale |
|---|---|
| **SiLU (Swish) activation** | Smoother than ReLU, non-monotonic; consistently outperforms ReLU on image classification — used throughout EfficientNet |
| **Residual connections** | Allow gradients to propagate cleanly to early layers; critical from scratch without pretraining |
| **Squeeze-Excitation** | Cheap channel attention (< 1% parameter overhead per block) that lets the network recalibrate feature importance per image |
| **Conv 7×7 stem** | Aggressive early downsampling (same as ResNet) reduces compute; the large receptive field in the first layer captures broad low-level features |
| **Global Average Pooling** | Eliminates fully-connected layers over spatial maps; reduces parameters and overfitting |
| **Kaiming initialisation** | Essential from scratch: fan-out mode with ReLU gain (the standard approximation for SiLU) ensures variance is preserved through depth |
| **Batch Normalisation everywhere** | Reduces internal covariate shift; makes the learning-rate schedule more forgiving |

### 4.4 Classical Baseline — HOG + SVM

**Source:** `pt_02_ml_baseline_svm.ipynb`

**Feature extraction:**
- Input: 128×128 grayscale image
- HOG: 9 orientations, 8×8 pixels/cell, 2×2 cells/block, L2-Hys block normalisation
- Output: fixed-length feature vector per image (~6,084 dims)

**Model:**
- `StandardScaler` → zero mean, unit variance
- RBF SVM (`sklearn.svm.SVC(probability=True)`)
- Tuned via `GridSearchCV`: `C ∈ {1, 10, 100}`, `gamma ∈ {scale, auto}`, 3-fold stratified CV, `f1_macro` scoring

**Why include it:** The SVM is genuinely useful as the performance floor and as a demonstration that classical handcrafted features can be competitive on a small, domain-specific dataset — particularly on the problematic official Kaggle split where the distribution shift hurts deep models more than HOG.

---

## 5. Training Pipeline

**Source:** `src/engine.py`, `train.py`

### 5.1 Two-Phase Progressive Fine-Tuning

Transfer-learning models use a two-phase schedule that avoids catastrophic forgetting of the pretrained weights.

```
Phase 1: Head-only training
─────────────────────────────────────────────────────
  freeze_backbone(model)         → all backbone params: requires_grad = False
  Optimizer: AdamW(head params, lr=1e-3, wd=1e-4)
  Schedule:  ReduceLROnPlateau(patience=3, factor=0.3)
  Duration:  12 epochs (+ early stopping, patience=7)

  Purpose: Bring the random head to a reasonable point before touching
  the pretrained features. Training the head at a high LR while the
  backbone is frozen is safe and fast.

Phase 2: Partial unfreeze + fine-tuning
─────────────────────────────────────────────────────
  unfreeze_top(model, frac=0.30) → top 30% of backbone: requires_grad = True
  Optimizer: AdamW with differential learning rates:
      backbone params: lr = 1e-5  (very small — preserve pretrained features)
      head params:     lr = 1e-4  (10× the backbone — head adapts faster)
  Schedule:  ReduceLROnPlateau(patience=3, factor=0.3)
  Duration:  25 epochs (+ early stopping)

  Purpose: Fine-tune the task-specific top layers at a LR small enough
  not to destroy ImageNet representations, while allowing the head to
  continue adapting more aggressively.
```

**Why 30% unfreeze?** Later layers encode the most task-specific, spatially local features — these benefit most from domain adaptation. Early layers encode generic edges and textures that transfer universally; retraining them on 3K medical images risks overfitting.

**Checkpoint saving:** The best-val-loss state dict is saved after each improvement. Phase 2 always resumes from the best Phase 1 checkpoint (via `model.load_state_dict(best_state)` at end of Phase 1).

### 5.2 From-Scratch Training Schedule

Custom CNN bypasses the two-phase schedule (no backbone to freeze):

```
All parameters trainable from random init
Optimizer: AdamW(all params, lr=3e-4, wd=1e-4)
Schedule:  ReduceLROnPlateau or CosineAnnealingWarmRestarts
Duration:  40 epochs (+ early stopping, patience=7)
MixUp:     enabled (alpha=0.2)
```

A more conservative `lr=3e-4` (vs. `1e-3` for the head-only Phase 1) is appropriate because all layers start from random init — too high an LR causes instability at depth.

### 5.3 Loss Functions

**Source:** `src/losses.py`

#### CrossEntropyLoss (default)

```python
nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
```

- `class_weights`: inverse-frequency tensor, counters the `notumor` class imbalance.
- `label_smoothing`: set to 0.0 in the final run (was tuned but found to hurt minority-class recall).

#### FocalLoss (optional, `--loss focal`)

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

- `γ = 2.0` (the standard focusing parameter)
- `α_t = class_weights` (same inverse-frequency weighting)
- `(1 - p_t)^γ` down-weights easy, confidently-classified examples so the optimiser focuses on the hard minority examples

**Note:** Focal loss is incompatible with MixUp (which produces soft targets). The engine detects this and silently disables MixUp when focal loss is selected.

### 5.4 Optimisation & Learning Rate Scheduling

**Optimiser:** AdamW throughout. Weight decay (`1e-4`) is applied to weight matrices only (AdamW's L2 regularisation is correct — unlike Adam with L2, AdamW keeps the adaptive gradient scaling and L2 penalty separate).

**Schedulers:**

| Option | Class | Step timing | Description |
|---|---|---|---|
| `plateau` (default) | `ReduceLROnPlateau` | Per epoch (on val loss) | Multiply LR by 0.3 after 3 epochs without improvement |
| `cosine` | `CosineAnnealingWarmRestarts` | Per batch (fractional epoch) | T0=10, Tmult=2; allows periodic LR restarts that escape local minima |

The cosine scheduler is stepped with a fractional epoch index after each batch:
```python
scheduler.step(epoch_idx + batch_i / n_batches)
```

### 5.5 Regularisation

| Technique | Where | Notes |
|---|---|---|
| **Dropout** (p=0.4 head, p=0.2 bottleneck) | All models — classifier head | Medical datasets are small; dropout is critical |
| **Dropout2d** (p=0.1) | CustomCNN residual blocks | Spatial channel dropout in deep layers |
| **Weight decay** (1e-4) | All models | AdamW — decoupled from gradient adaptation |
| **MixUp** (α=0.2) | CustomCNN training | Convex combination of samples + soft labels; improves generalisation on small datasets |
| **Early stopping** (patience=7, min_delta=1e-4) | All models | Prevents overfitting; best checkpoint is always restored |
| **Augmentation** | Training only | See transform pipeline below |

**Augmentation pipeline (training):**

```
MRIPreprocess (brain-crop + CLAHE)
→ RandomResizedCrop(224, scale=(0.9, 1.0))
→ RandomHorizontalFlip(p=0.5)
→ RandomRotation(±15°)
→ RandomAffine(translate=(0.08, 0.08))
→ ColorJitter(brightness=0.1, contrast=0.1)
→ ToTensor → Normalise(ImageNet stats)
```

**Medical safety:** Vertical flips and heavy geometric distortions are deliberately excluded — they create anatomically impossible scans and confuse a model trained to distinguish spatial tumour locations.

### 5.6 Automatic Mixed Precision (AMP)

```python
scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
with torch.amp.autocast("cuda", enabled=cfg.use_amp):
    logits = model(x)
    loss = criterion(logits, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

FP16 forward pass + FP32 gradient accumulation. Enabled automatically when a CUDA GPU is available. On an RTX 3060 this approximately **halves training time** and reduces peak VRAM by ~30%.

---

## 6. Evaluation Suite

**Source:** `src/evaluate.py`

| Metric | Implementation |
|---|---|
| **Accuracy** | `sklearn.metrics.classification_report` |
| **Macro-F1** | Unweighted average across 4 classes — primary metric (robust to class imbalance) |
| **Weighted-F1** | Support-weighted average |
| **ROC-AUC** | One-vs-rest macro AUC (`roc_auc_score`, `multi_class='ovr'`) |
| **ECE** | Expected Calibration Error — 15 equal-width bins |
| **Inference latency** | Mean ± std ms/image over 20 timed batches (CUDA-synchronised) |
| **Confusion matrix** | Raw counts + row-normalised recall heatmap |
| **ROC curves** | Per-class + macro, saved to `results/` |
| **McNemar's test** | Paired significance test on same test set (see §9) |

**Test-Time Augmentation (TTA):** When `cfg.use_tta=True`, each test image is passed through the model twice — original and horizontal flip — and the softmax probabilities are averaged. This gives a cheap ~0.5% accuracy boost with no extra training.

---

## 7. Explainability — Grad-CAM

**Source:** `app/inference.py`, `pt_04_gradcam_inference.ipynb`

Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017), implemented from scratch in PyTorch without external libraries.

### Algorithm

```
1. Target layer: the last Conv2d in the network (most semantically rich, still spatial)

2. Forward hook on target layer:
       store["activations"] = output  # (B, C, H', W')

3. Backward hook on target layer:
       store["gradients"] = grad_output[0]  # (B, C, H', W')

4. Forward pass → logits → argmax (the predicted class c*)
   model.zero_grad(); logits[0, c*].backward()

5. Weights: α_c = mean_{H',W'}(∂y_{c*} / ∂A_c^k)   # (C,)
           = GAP over the gradient maps

6. Heatmap: L = ReLU( Σ_c α_c · A_c )               # (H', W')
             → normalise to [0, 1]
             → resize to 224×224

7. Overlay: JET colormap(heatmap) blended at 0.45 over
            the preprocessed MRI at 0.55
```

### Implementation Notes

- Hooks are registered immediately before each inference call and always removed in a `finally` block — this prevents hooks from accumulating on a cached model.
- `register_full_backward_hook` is used (not the deprecated `register_backward_hook`) so the gradient tensor has the correct shape `(B, C, H', W')` rather than the summed-over-batch form.
- The `requires_grad_(True)` call on the input tensor is required for the backward pass through a `torch.no_grad()`-cached model.

### What the maps show

| Class | Clinically expected attention |
|---|---|
| Glioma | Irregular infiltrating mass, often frontal/temporal lobes |
| Meningioma | Dural attachment, extra-axial location near skull |
| Pituitary | Sella turcica (pituitary fossa), midline structure |
| No Tumor | Diffuse, unfocused — no single anatomical hotspot |

The model learns these spatial biases from class labels alone, with no anatomical supervision.

---

## 8. Post-Hoc Calibration

**Source:** `src/models.py` — `TemperatureScaling`, `ece_score`

Deep neural networks are systematically over-confident: a prediction of "95% glioma" often corresponds to a real accuracy well below 95%. In a medical context, confidence values may be read by clinicians, so calibration matters independently of accuracy.

### Temperature Scaling

A single scalar parameter `T` is fitted on the validation set (not the test set):

```
calibrated_logits = raw_logits / T
```

`T` is found by minimising NLL on the validation set using LBFGS (typically converges in < 50 iterations). Setting `T > 1` softens the softmax distribution; `T < 1` sharpens it.

**Properties:**
- Does not change the argmax (accuracy is untouched)
- Operates on logits after training — the base model weights are frozen
- One parameter → no risk of overfitting on the val set

### Expected Calibration Error (ECE)

```
ECE = Σ_b (|B_b| / N) × | acc(B_b) - conf(B_b) |
```

Predictions are binned by confidence into 15 equal-width bins. For each bin, the gap between mean accuracy and mean confidence is weighted by the fraction of samples in that bin. A perfectly calibrated model has ECE = 0.

**Results:** Custom CNN raw ECE ≈ 0.05, temperature ≈ 1.06 — already nearly calibrated out of the box. The small temperature (barely above 1) confirms the model is slightly over-confident but not dramatically so.

---

## 9. Statistical Significance Testing

**Source:** `src/evaluate.py` — `mcnemar_test`

McNemar's test determines whether two models' error rates on the **same** test set are significantly different (paired test, unlike a t-test on independent samples).

### Contingency Table

```
                    Model B correct   Model B wrong
Model A correct         n11               n10
Model A wrong           n01               n00
```

Only the **discordant** cells matter (`n10`, `n01`). Under H₀ (equal error rates), they are equally likely.

### Test Statistic

```
χ² = (|n10 - n01| - 1)² / (n10 + n01)      [with Yates' continuity correction]
```

Distributed as χ² with 1 degree of freedom. A `p-value < 0.05` indicates the models are significantly different on this test set.

**Usage in `pt_06`:** The best-performing model (Custom CNN) is tested against each other model. The null hypothesis is that their predictions are interchangeable.

---

## 10. Live Demo Application

**Source:** `app/main.py`, `app/inference.py`, `app/static/index.html`

### Architecture

```
Browser
  │  POST /api/predict  (multipart: model=..., file=<image>)
  │
  ▼
FastAPI (app/main.py)
  │
  ▼
inference.predict(model_name, image_bytes)
  │
  ├── load_model(name)          lazy-load + cache
  │      │
  │      ├── deep models:  build_model(name) → load_state_dict → .eval()
  │      └── svm_hog:      joblib.load(svm + scaler + label_encoder)
  │
  ├── PIL.open → preprocess (same MRIPreprocess as training)
  │
  ├── deep:  _deep_probs()      AMP-free forward → softmax
  │          _gradcam_overlay() forward+backward hooks → heatmap → base64 PNG
  │
  └── svm:   _svm_probs()       HOG → scaler.transform → svm.predict_proba
```

### Model Caching

Models are loaded on first selection and cached in a module-level dict. Subsequent predictions against the same model are instant. The cache is process-scoped (per `uvicorn` worker) — restarting the server clears it.

### Grad-CAM in the Demo

The overlay path is the same algorithm as §7 but packaged for single-image real-time use. The heatmap is returned as a base64-encoded PNG data URI embedded in the JSON response so no temporary files are required.

### API Reference

| Method | Endpoint | Payload | Response |
|---|---|---|---|
| `GET` | `/` | — | HTML page |
| `GET` | `/api/models` | — | `{models: [{id, label, type, params}], device}` |
| `POST` | `/api/load` | `model=<id>` (form) | `{status, model}` |
| `POST` | `/api/predict` | `model=<id>`, `file=<image>` (multipart) | `{pred_label, confidence, scores, overlay, explainable, device}` |

---

## 11. Configuration Reference

**Source:** `src/config.py`

All hyperparameters live in a single `Config` dataclass. Every run serialises the full config to `results/run_config.json`, making experiments fully reproducible.

| Parameter | Default | Description |
|---|---|---|
| `seed` | 42 | Global random seed (Python, NumPy, PyTorch, cuDNN) |
| `img_size` | 224 | Input resolution (H × W) |
| `batch_size` | 32 | Samples per batch |
| `num_workers` | 0 | DataLoader workers (0 = safest on Windows) |
| `val_split` | 0.15 | Fraction of total data for validation |
| `test_split` | 0.15 | Fraction of total data for test |
| `resplit_data` | `True` | Pool all images and stratified re-split (Protocol B) |
| `use_crop` | `True` | Brain-region Otsu crop |
| `use_clahe` | `True` | CLAHE contrast enhancement |
| `clahe_clip_limit` | 2.0 | CLAHE amplification cap |
| `clahe_grid` | (8, 8) | CLAHE tile grid size |
| `aug_rotation_deg` | 15 | Max random rotation angle (degrees) |
| `aug_translate` | 0.08 | Max random translation (fraction of image) |
| `aug_zoom` | (0.9, 1.0) | Random crop scale range |
| `aug_brightness` | 0.10 | ColorJitter brightness factor |
| `aug_contrast` | 0.10 | ColorJitter contrast factor |
| `aug_hflip_p` | 0.5 | Horizontal flip probability |
| `label_smoothing` | 0.0 | CrossEntropy label smoothing (0 = disabled) |
| `weight_decay` | 1e-4 | AdamW L2 regularisation |
| `use_class_weights` | `True` | Inverse-frequency loss weighting |
| `use_amp` | auto | Automatic mixed precision (True when CUDA available) |
| `loss_type` | `cross_entropy` | `"cross_entropy"` or `"focal"` |
| `focal_gamma` | 2.0 | Focal loss focusing parameter γ |
| `lr_scheduler` | `plateau` | `"plateau"` or `"cosine"` |
| `use_mixup` | `False` | MixUp augmentation (Beta(α, α) mixing) |
| `mixup_alpha` | 0.2 | MixUp mixing coefficient |
| `scratch_epochs` | 40 | Training epochs for from-scratch models |
| `scratch_lr` | 3e-4 | Learning rate for from-scratch training |
| `phase1_epochs` | 12 | Phase 1 (head-only) epochs |
| `phase1_lr` | 1e-3 | Phase 1 learning rate |
| `phase2_epochs` | 25 | Phase 2 (partial unfreeze) epochs |
| `phase2_lr` | 1e-5 | Phase 2 backbone learning rate (head = 10×) |
| `unfreeze_fraction` | 0.30 | Fraction of backbone params to unfreeze in Phase 2 |
| `early_stop_patience` | 7 | Epochs without val-loss improvement before stopping |
| `early_stop_min_delta` | 1e-4 | Minimum val-loss improvement to count as progress |
| `lr_plateau_patience` | 3 | ReduceLROnPlateau patience |
| `lr_plateau_factor` | 0.3 | ReduceLROnPlateau LR multiplier |
| `dropout_head` | 0.4 | Dropout probability in the classifier head |
| `use_tta` | `True` | Test-time augmentation (horizontal flip average) |

---

*Document generated from the source code. For the latest parameter values, `src/config.py` is authoritative.*
