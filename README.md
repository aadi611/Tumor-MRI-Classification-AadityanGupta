# Brain Tumor MRI Classification

**Capstone Project — IIT Roorkee PG Diploma in Data Science & AI**
**Author:** Aadityan Gupta · [GitHub](https://github.com/aadi611)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![FastAPI](https://img.shields.io/badge/FastAPI-demo-009688)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

A 4-class brain-tumor MRI classifier that benchmarks a classical **HOG + SVM** baseline against a portfolio of transfer-learned CNNs — **EfficientNet-B0, ResNet-50, DenseNet-121, MobileNetV3** — and a **custom residual SE-CNN trained from scratch**, with **Grad-CAM** visual explainability, post-hoc temperature-scaling calibration, and a live **FastAPI** inference demo.

---

## Table of Contents

- [Highlights](#highlights)
- [Results](#results)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [Live Demo](#live-demo)
- [How It Works](#how-it-works)
- [References](#references)

---

## Highlights

| Feature | Detail |
|---|---|
| **Classical baseline** | HOG (9-orient, 8×8 cells) + RBF SVM tuned by GridSearchCV — sets the performance floor |
| **Transfer learning** | EfficientNet-B0, ResNet-50, DenseNet-121, MobileNetV3 — two-phase progressive fine-tuning |
| **From scratch** | Custom residual SE-CNN with SiLU activations (~11.3 M params) — a clean ablation of architecture vs. pretraining |
| **Explainability** | Grad-CAM via pure-PyTorch forward/backward hooks; verifies attention on clinically relevant anatomy |
| **Calibration** | Post-hoc temperature scaling (Guo et al., 2017) + Expected Calibration Error (ECE) measurement |
| **Statistical test** | McNemar's test for paired model significance |
| **Modular pipeline** | Config-driven `src/` package: AMP, early stopping, MixUp, focal loss, cosine warm restarts |
| **Live demo app** | FastAPI backend with lazy model loading + Grad-CAM overlay; `python run_demo.py` to launch |

---

## Results

> A key finding: **the official Kaggle train/test split has a documented distribution shift** that severely penalises deep models on the `glioma` class. Results are therefore reported under two evaluation protocols — and the gap between them is itself an important result.

### Protocol A — Official Kaggle `Testing/` Folder

Evaluated in `pt_02` and `pt_03`.

| Model | Accuracy | Macro-F1 | ROC-AUC | Glioma Recall |
|---|---|---|---|---|
| HOG + SVM (baseline) | **0.772** | **0.728** | 0.946 | — |
| EfficientNet-B0 (transfer) | 0.632 | 0.585 | 0.861 | **0.16** |

On this split, EfficientNet collapses on glioma (16% recall). This is an **artifact of the distribution shift**, not evidence the classical baseline is genuinely better — the Training and Testing glioma subsets come from different distributions.

### Protocol B — Pooled Stratified 70 / 15 / 15 Re-split

Evaluated by the `src/` pipeline with `resplit_data=True` (the default).

| Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| Custom CNN (from scratch) | **0.904** | **0.906** | **0.988** |
| ResNet-50 | 0.894 | 0.903 | 0.988 |
| EfficientNet-B0 | ~0.890 | ~0.900 | ~0.982 |

When train and test are drawn from the same distribution, deep models recover to 89–90% and comfortably outperform the baseline. The from-scratch Custom CNN achieves **glioma recall of 0.86** (vs 0.16 under Protocol A) with all four classes landing in the 0.86–0.94 F1 range — balanced with no weak class.

**Calibration:** Custom CNN — temperature ≈ 1.06, ECE ≈ 0.05 (already well-calibrated out of the box).

> **Key takeaway:** evaluation methodology matters as much as the model. The `src/` pipeline defaults to Protocol B precisely because it reflects true generalisation.

---

## Dataset

**Source:** [sartajbhuvaji/brain-tumor-classification-mri](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) on Kaggle (~3,264 training / ~394 test images).

| Class | Training | Testing |
|---|---|---|
| Glioma | 826 | 100 |
| Meningioma | 822 | 115 |
| No Tumor | 395 | 105 |
| Pituitary | 827 | 74 |

**Class imbalance:** `No Tumor` has roughly half the samples of the other classes. This is addressed through **class-weighted loss**, **stratified splits**, and **macro-F1** as the primary metric.

---

## Repository Structure

```
Brain-Tumor-Improved/
│
├── README.md                         # This file
├── ARCHITECTURE.md                   # Deep-dive into every model and training design decision
├── requirements.txt                  # Python dependencies
├── train.py                          # CLI training orchestrator
├── run_demo.py                       # One-command demo launcher (FastAPI + browser)
│
├── pt_01_eda.ipynb                   # EDA: class distribution, sample images, t-SNE
├── pt_02_ml_baseline_svm.ipynb       # HOG + SVM baseline with GridSearchCV
├── pt_03_efficientnet_training.ipynb # EfficientNet-B0 two-phase transfer learning
├── pt_04_gradcam_inference.ipynb     # Grad-CAM explainability + single-image inference
├── pt_05_custom_cnn.ipynb            # From-scratch custom CNN training & evaluation
├── pt_06_comparison.ipynb            # Cross-model comparison dashboard
│
├── src/                              # Engineered, config-driven pipeline package
│   ├── __init__.py                   #   Public API re-exports
│   ├── config.py                     #   All hyperparameters in one dataclass (Config)
│   ├── preprocessing.py              #   MRI-specific: Otsu brain-crop + CLAHE
│   ├── data.py                       #   Dataset, stratified re-split, dataloaders, class weights
│   ├── losses.py                     #   CrossEntropy / FocalLoss factory
│   ├── models.py                     #   Model factory, freeze/unfreeze, temperature scaling, ECE
│   ├── custom_cnn.py                 #   From-scratch residual SE-CNN architecture
│   ├── engine.py                     #   Training loops: AMP, early stop, two-phase, MixUp, cosine
│   ├── evaluate.py                   #   Metrics, ROC, Grad-CAM deep-dive, McNemar, latency
│   └── utils.py                      #   Seeding, config logging, split reporting
│
├── app/                              # FastAPI live demo
│   ├── main.py                       #   REST endpoints: /api/models, /api/load, /api/predict
│   ├── inference.py                  #   Lazy model cache, Grad-CAM overlay, predict()
│   └── static/index.html             #   Single-page UI (dropdown, upload, results panel)
│
└── results/                          # Generated figures and metrics (committed for review)
    ├── leaderboard.json
    ├── *_confusion_matrix.png
    ├── *_roc.png
    ├── *_training_curves.png
    └── ...
```

> **Not in the repo** (gitignored): `data/`, `models/*.pth`, `models/*.pkl`, `_local/`.
> Regenerate locally by running the notebooks or `python train.py --compare`.

---

## Setup

**Prerequisites:** Python 3.11 · NVIDIA GPU strongly recommended (CPU works but training is several hours)

### 1. Clone the repository

```bash
git clone https://github.com/aadi611/Tumor-MRI-Classification-AadityanGupta.git
cd Tumor-MRI-Classification-AadityanGupta
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install PyTorch (GPU build)

```bash
# Adjust the CUDA version suffix to match your driver (cu124 = CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

For CPU-only:

```bash
pip install torch torchvision
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the dataset

```bash
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri
unzip brain-tumor-classification-mri.zip -d data/
```

### 6. Rename dataset folders

The code expects short folder names (`glioma`, `meningioma`, `notumor`, `pituitary`). Rename them:

**Windows (PowerShell):**

```powershell
# Training
cd data\Training
mv glioma_tumor   glioma
mv meningioma_tumor meningioma
mv no_tumor       notumor
mv pituitary_tumor pituitary

# Testing
cd ..\Testing
mv glioma_tumor   glioma
mv meningioma_tumor meningioma
mv no_tumor       notumor
mv pituitary_tumor pituitary
cd ..\..
```

**macOS / Linux:**

```bash
for split in Training Testing; do
  cd data/$split
  mv glioma_tumor glioma; mv meningioma_tumor meningioma
  mv no_tumor notumor; mv pituitary_tumor pituitary
  cd ../..
done
```

---

## Running the Project

### Option A — Notebooks (run in order)

| Notebook | What it does |
|---|---|
| `pt_01_eda.ipynb` | Explore the dataset: distribution, sample images, mean-image per class, t-SNE separability |
| `pt_02_ml_baseline_svm.ipynb` | HOG feature extraction → SVM with GridSearchCV. Saves `models/svm_*.pkl` |
| `pt_03_efficientnet_training.ipynb` | Phase 1 (frozen, 15 ep) → Phase 2 (fine-tune, 20 ep). Saves `models/efficientnet_best.pth` |
| `pt_04_gradcam_inference.ipynb` | Grad-CAM overlays on correctly classified and misclassified examples |
| `pt_05_custom_cnn.ipynb` | Train and evaluate the from-scratch custom CNN |
| `pt_06_comparison.ipynb` | Cross-model leaderboard, Pareto frontier, calibration and latency comparison |

### Option B — Command-line pipeline

```bash
# Train a single model (defaults: resplit_data=True, cross-entropy loss)
python train.py --model resnet50
python train.py --model efficientnet_b0
python train.py --model densenet121
python train.py --model custom_cnn

# Train all models and write leaderboard.json
python train.py --compare

# Use the original Kaggle split (reproduces Protocol A results)
python train.py --model efficientnet_b0 --no-resplit

# Experiment with focal loss
python train.py --model resnet50 --loss focal

# Override training epochs
python train.py --model resnet50 --epochs1 10 --epochs2 20
```

**Approximate training time** (RTX 3060, mixed precision):

| Model | Time |
|---|---|
| EfficientNet-B0 | ~25 min |
| ResNet-50 | ~35 min |
| DenseNet-121 | ~30 min |
| Custom CNN | ~40 min |
| HOG + SVM | ~10 min |
| CPU (any model) | Several hours |

---

## Live Demo

The FastAPI demo app lets you upload any brain MRI image and classify it with any trained model — displaying per-class confidence scores and a Grad-CAM overlay for deep models.

```bash
python run_demo.py
# → opens http://127.0.0.1:8000 automatically
```

**How it works:**
- Models are lazy-loaded and cached on first selection (~1–3 s warm-up; instant thereafter).
- Grad-CAM is computed in real time via forward/backward hooks on the last conv layer.
- The SVM baseline is also available (no Grad-CAM — it is not a CNN).
- The app reuses the exact same `src/` preprocessing the models were trained with, so inference predictions match the evaluation pipeline.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Single-page UI |
| `GET` | `/api/models` | List available models (drives the dropdown) |
| `POST` | `/api/load` | Warm-load a model into memory |
| `POST` | `/api/predict` | Classify an uploaded image |

---

## How It Works

### Preprocessing (`src/preprocessing.py`)

Every image passes through two domain-specific steps before augmentation:

1. **Brain-region crop** — Otsu threshold → morphological clean-up → largest external contour → tight bounding box. Removes the large black border around the skull so the full input resolution is spent on anatomy.
2. **CLAHE** — Contrast-Limited Adaptive Histogram Equalisation on the L-channel (LAB colour space). Normalises local contrast across different scanners and MRI sequences without blowing out noise.

### Classical Baseline (`pt_02`)

HOG features (9 orientations, 8×8 pixels/cell, 2×2 cells/block, L2-Hys normalisation) on 128×128 grayscale images → `StandardScaler` → RBF **SVM** with `GridSearchCV` (3-fold stratified CV, `f1_macro` scoring).

### Transfer Learning (`pt_03`, `src/`)

ImageNet-pretrained backbone with a fresh classification head (`in_features → Dropout → 256 → BN → ReLU → Dropout → 4`), trained in two phases:

- **Phase 1** — backbone frozen, head only, LR 1e-3, 12 epochs.
- **Phase 2** — top 30% of backbone unfrozen, differential LRs (backbone: 1e-5, head: 1e-4), 25 epochs.

Additional techniques: class-weighted cross-entropy, AMP (automatic mixed precision), `ReduceLROnPlateau`, early stopping, best-checkpoint saving.

### Custom CNN (`pt_05`, `src/custom_cnn.py`)

A from-scratch network with no pretrained weights, designed as a control experiment isolating architecture from pretraining:

- **Stem** — Conv 7×7/2 → BN → SiLU → MaxPool 3×3/2
- **4 Residual SE stages** — each with 2 × `ResidualSEBlock` (3×3 convs + BatchNorm + SiLU + Squeeze-Excitation attention)
- **Head** — Global Average Pooling → Dropout → Linear(512 → 4)
- ~11.3 M parameters · trained with cosine warm restarts + MixUp

### Explainability (`pt_04`, `app/inference.py`)

**Grad-CAM** via pure-PyTorch forward and full-backward hooks on the last Conv2d layer:
1. Forward hook captures activation maps `A` of shape `(C, H, W)`.
2. Backward hook captures gradients `∂y_c / ∂A` for the predicted class.
3. Weights = global-average-pooled gradients → `(C,)`.
4. Heatmap = `ReLU( Σ_c weight_c · A_c )`, resized to 224×224 and blended over the preprocessed MRI.

### Calibration (`src/models.py`)

Post-hoc temperature scaling: a single scalar `T` fitted on the validation set via LBFGS (minimising NLL). Divides the model's logits by `T` before softmax — does not change predictions, only confidence values. Custom CNN achieved `T ≈ 1.06`, ECE ≈ 0.05.

---

## References

1. Sartaj Bhuvaji et al. — *Brain Tumor Classification (MRI)*. Kaggle, 2020.
2. Tan, M. & Le, Q.V. — *EfficientNet: Rethinking Model Scaling for CNNs*. ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
3. He, K. et al. — *Deep Residual Learning for Image Recognition*. CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
4. Huang, G. et al. — *Densely Connected Convolutional Networks*. CVPR 2017. [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)
5. Selvaraju, R.R. et al. — *Grad-CAM: Visual Explanations via Gradient-based Localization*. ICCV 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
6. Hu, J. et al. — *Squeeze-and-Excitation Networks*. CVPR 2018. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
7. Guo, C. et al. — *On Calibration of Modern Neural Networks*. ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
8. Zhang, H. et al. — *MixUp: Beyond Empirical Risk Minimization*. ICLR 2018. [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
9. Lin, T.-Y. et al. — *Focal Loss for Dense Object Detection*. ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
10. Dalal, N. & Triggs, B. — *Histograms of Oriented Gradients for Human Detection*. CVPR 2005.
