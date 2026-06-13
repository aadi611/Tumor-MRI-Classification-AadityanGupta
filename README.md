# Brain Tumor MRI Classification

**Capstone Project — IIT Roorkee PG Diploma in Data Science & AI**
**Author:** Aadityan Gupta

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![Status](https://img.shields.io/badge/status-complete-success)

A 4-class brain-tumor classifier from MRI scans (**glioma · meningioma · pituitary · no-tumor**) that benchmarks a classical **HOG + SVM** baseline against transfer-learned CNNs (**EfficientNet-B0**, ResNet-50, DenseNet-121, MobileNetV3) and a **from-scratch custom CNN**, with **Grad-CAM** explainability and a full evaluation suite (calibration, statistical significance, latency).

Repo: [aadi611/Tumor-MRI-Classification-AadityanGupta](https://github.com/aadi611/Tumor-MRI-Classification-AadityanGupta)

---

## Highlights

- **Classical → deep comparison** — a tuned HOG + SVM baseline establishes the performance floor; transfer learning and a custom CNN quantify the gain.
- **Two-phase transfer learning** — freeze the ImageNet backbone to train a new head, then fine-tune the top layers at a 100× smaller LR (with differential learning rates).
- **From-scratch custom CNN** — a residual + squeeze-excitation + SiLU network as a *control experiment* for "architecture vs. pretraining."
- **Grad-CAM from scratch** — pure-PyTorch forward/backward hooks; verifies the model attends to correct anatomy.
- **Production-grade `src/` pipeline** — a modular, config-driven, CLI-runnable package (not just notebooks): brain-crop + CLAHE preprocessing, stratified re-split, AMP, MixUp, cosine restarts, focal loss, temperature-scaling calibration, McNemar significance testing, and inference-latency benchmarking.
- **Comparison dashboard** — leaderboard, efficiency (Pareto) frontier, and calibration comparison across all models.

---

## Results

Two evaluation protocols are used:
- **Notebooks `pt_02`–`pt_04`** evaluate on the **original Kaggle `Testing/` folder**.
- **The modular `src/` pipeline** (`resplit_data=True`) pools all images and re-splits **stratified 70/15/15** to remove a documented distribution shift and measure honest generalization (test set n = 490).

| Model | Protocol | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|---|
| HOG + SVM (baseline) | original test | **0.772** | **0.728** | 0.946 |
| EfficientNet-B0 (transfer) | original test | **~0.89** | **~0.90** | ~0.98 |
| ResNet-50 (portfolio best) | stratified test | **0.894** | **0.903** | **0.988** |
| Custom CNN (from scratch) | stratified test | **0.908** | ~0.89 | ~0.96 |

> Figures for SVM, ResNet-50, and the Custom CNN are exact (from `results/baseline_results.json` and `results/leaderboard.json`). EfficientNet-B0 figures are the notebook's reported range; reproduce any number via the notebooks or `python train.py --compare`.

**Calibration:** the custom CNN fit a temperature of ≈1.06 with ECE ≈ 0.05 — already well-calibrated out of the box.

---

## Dataset

Source: [sartajbhuvaji/brain-tumor-classification-mri](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) on Kaggle (~3,264 train / ~394 test images).

| Class | Train | Test |
|---|---|---|
| Glioma | 826 | 100 |
| Meningioma | 822 | 115 |
| No Tumor | 395 | 105 |
| Pituitary | 827 | 74 |

**Class imbalance:** `No Tumor` has ~half the samples of the other classes — handled with **class-weighted loss**, **stratified splits**, and **macro-F1** as the primary metric.

---

## Repository Structure

```
Tumor-MRI-Classification-AadityanGupta/
├── README.md
├── requirements.txt
├── train.py                          # CLI orchestrator (train one model or --compare the portfolio)
│
├── pt_01_eda.ipynb                   # EDA: distribution, samples, intensity, mean-image, t-SNE
├── pt_02_ml_baseline_svm.ipynb       # HOG + SVM baseline (GridSearchCV, stratified CV)
├── pt_03_efficientnet_training.ipynb # EfficientNet-B0 — 2-phase transfer learning
├── pt_04_gradcam_inference.ipynb     # Grad-CAM explainability + inference demo
├── pt_05_custom_cnn.ipynb            # From-scratch custom CNN (residual + SE + SiLU)
├── pt_06_comparison.ipynb            # Cross-model comparison dashboard
│
├── src/                              # The engineered pipeline
│   ├── config.py                     #   all hyperparameters (one dataclass)
│   ├── preprocessing.py              #   brain-region crop (Otsu) + CLAHE
│   ├── data.py                       #   dataset, stratified re-split, dataloaders, class weights
│   ├── losses.py                     #   CrossEntropy / Focal loss factory
│   ├── models.py                     #   model factory, freeze/unfreeze, temperature scaling, ECE
│   ├── engine.py                     #   training loops: AMP, early-stop, 2-phase, MixUp, cosine
│   ├── evaluate.py                   #   metrics, ROC, Grad-CAM deep-dive, McNemar, latency
│   └── utils.py                      #   seeding, config logging, split reporting
│
└── results/                          # Figures, leaderboard.json, predictions (committed for review)
```

Not in the repo (gitignored — regenerate locally): `data/`, `models/` (`*.pth`, `*.pkl`), `*.zip`, `.venv/`.

---

## Setup

**Prerequisites:** Python 3.11; NVIDIA GPU recommended (CPU works but is slow).

```bash
# 1. Clone
git clone https://github.com/aadi611/Tumor-MRI-Classification-AadityanGupta
cd Tumor-MRI-Classification-AadityanGupta

# 2. Virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Install PyTorch first (GPU build — adjust CUDA version to your machine)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Install the rest
pip install -r requirements.txt

# 5. Download the dataset from Kaggle
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri
unzip brain-tumor-classification-mri.zip -d data/
```

**Rename dataset folders** to the short class names the code expects (`glioma`, `meningioma`, `notumor`, `pituitary`):

```powershell
# Windows / PowerShell
cd data\Training; mv glioma_tumor glioma; mv meningioma_tumor meningioma; mv no_tumor notumor; mv pituitary_tumor pituitary; cd ..\Testing; mv glioma_tumor glioma; mv meningioma_tumor meningioma; mv no_tumor notumor; mv pituitary_tumor pituitary; cd ..\..
```

---

## Running the Project

### Option A — the notebooks (run in order)

1. `pt_01_eda.ipynb` — explore the data (distribution, samples, mean-image, t-SNE separability).
2. `pt_02_ml_baseline_svm.ipynb` — HOG → SVM with GridSearchCV. Saves `models/svm_model.pkl`.
3. `pt_03_efficientnet_training.ipynb` — Phase 1 (frozen, 15 ep) → Phase 2 (fine-tune, 20 ep). Saves `models/efficientnet_best.pth`.
4. `pt_04_gradcam_inference.ipynb` — Grad-CAM overlays + single-image inference demo.
5. `pt_05_custom_cnn.ipynb` — train & evaluate the from-scratch custom CNN.
6. `pt_06_comparison.ipynb` — build the cross-model comparison dashboard.

### Option B — the command-line pipeline

```bash
python train.py --model efficientnet_b0        # train one model
python train.py --compare                       # train the portfolio + write leaderboard.json
python train.py --model custom_cnn              # from-scratch CNN
python train.py --model densenet121 --loss focal --no-resplit
```

**Training time** (RTX 3060, mixed precision): ~25–40 min for the two-phase EfficientNet; CPU is several hours.

---

## How It Works

### Classical baseline (`pt_02`)
HOG (9 orientations, 8×8 cells, 2×2 blocks, L2-Hys) on 128×128 grayscale → `StandardScaler` → RBF **SVM** tuned by `GridSearchCV` (stratified 3-fold, `f1_macro` scoring).

### Deep model (`pt_03`, `src/`)
EfficientNet-B0 (ImageNet) with a custom head (`1280 → Dropout → 256 → ReLU → Dropout → 4`), trained in **two phases**: frozen backbone @ LR 1e-3, then top-30%-unfrozen fine-tuning @ LR 1e-5. Uses class-weighted cross-entropy, label smoothing, AMP, `ReduceLROnPlateau`, early stopping, and best-checkpoint saving.

### Custom CNN (`pt_05`)
A from-scratch network: Conv stem → 4 stages of **Residual + Squeeze-Excitation** blocks with **SiLU** activations → **Global Average Pooling** → linear head. ~11.3 M params, trained with cosine warm restarts + MixUp. A control experiment isolating *architecture* from *pretraining*.

### Explainability (`pt_04`)
**Grad-CAM** via pure-PyTorch forward/backward hooks on the last conv layer (`model.features[-1][0]`). The model attends to clinically meaningful anatomy — tumor mass (glioma), dural attachment (meningioma), sella turcica (pituitary) — learned from labels alone.

### Engineered pipeline (`src/`)
Adds: Otsu **brain-crop + CLAHE** preprocessing, stratified **re-split**, **focal loss**, **MixUp**, **cosine warm restarts**, **differential learning rates**, **temperature-scaling calibration + ECE**, **McNemar's significance test**, and **inference-latency benchmarking** — all config-driven via `src/config.py`.

---

## References

1. Sartaj Bhuvaji et al. — *Brain Tumor Classification (MRI)*. Kaggle, 2020.
2. Tan, M. & Le, Q.V. — *EfficientNet: Rethinking Model Scaling for CNNs*. ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
3. Selvaraju, R.R. et al. — *Grad-CAM: Visual Explanations via Gradient-based Localization*. ICCV 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
4. Hu, J. et al. — *Squeeze-and-Excitation Networks*. CVPR 2018. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
5. Guo, C. et al. — *On Calibration of Modern Neural Networks*. ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
6. Dalal, N. & Triggs, B. — *Histograms of Oriented Gradients for Human Detection*. CVPR 2005.
