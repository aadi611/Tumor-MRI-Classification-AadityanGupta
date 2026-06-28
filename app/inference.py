"""
inference.py
============
Loads the project's trained checkpoints on demand and runs single-image
prediction — reusing the exact `src/` preprocessing the models were trained with,
so demo predictions match the real pipeline.

Models are lazy-loaded and cached: the first time a model is selected it loads
(~1-3 s), and every prediction after that is instant.
"""
from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Silence the harmless "model pickled with a different scikit-learn version" warning.
# Only the classical HOG+SVM baseline triggers it; the deep models are unaffected.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

# Make the project root importable so `import src` works no matter where we launch from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import base64                                             # noqa: E402
import cv2                                                # noqa: E402
import torch.nn as nn                                     # noqa: E402

from src import Config, build_model                       # noqa: E402
from src.config import CLASSES, CLASS_LABELS, MODELS_DIR  # noqa: E402
from src.data import build_transforms                     # noqa: E402
from src.preprocessing import MRIPreprocess               # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG = Config()
_, EVAL_TF = build_transforms(CFG)        # MRIPreprocess(crop+CLAHE) -> Resize -> Normalize
# The same crop+CLAHE the model saw — the base image we paint the Grad-CAM heatmap onto.
_PRE = MRIPreprocess(use_crop=CFG.use_crop, use_clahe=CFG.use_clahe,
                     clahe_clip_limit=CFG.clahe_clip_limit, clahe_grid=CFG.clahe_grid)

# Deep checkpoints (architecture built by src.build_model) → file in models/
DEEP_FILES = {
    "custom_cnn":      "custom_cnn_best.pth",
    "resnet50":        "resnet50_best.pth",
    "efficientnet_b0": "efficientnet_b0_best.pth",
    "vgg16":           "vgg16_best.pth",
    "densenet121":     "densenet121_best.pth",
}
SVM_FILES = ("svm_model.pkl", "svm_scaler.pkl", "label_encoder.pkl")

# Display metadata for the dropdown (Protocol-B stratified test results).
META = {
    "custom_cnn":      {"label": "Custom CNN — from scratch ★",     "type": "From scratch",       "params": "22M"},
    "resnet50":        {"label": "ResNet-50 — transfer",            "type": "Transfer learning", "params": "23.8M"},
    "efficientnet_b0": {"label": "EfficientNet-B0 — efficient ◆",   "type": "Transfer learning", "params": "4.4M"},
    "vgg16":           {"label": "VGG-16 — dominated ▼",            "type": "Transfer learning", "params": "134.6M"},
    "densenet121":     {"label": "DenseNet-121 — transfer",         "type": "Transfer learning", "params": "7.3M"},
    "svm_hog":         {"label": "HOG + SVM — classical baseline",  "type": "Classical ML",      "params": "—"},
}
ORDER = ["custom_cnn", "resnet50", "efficientnet_b0", "densenet121", "vgg16", "svm_hog"]

_cache: dict = {}            # name -> loaded object
_skimage_ok: bool | None = None


def _has_skimage() -> bool:
    global _skimage_ok
    if _skimage_ok is None:
        try:
            import skimage.feature  # noqa: F401
            _skimage_ok = True
        except Exception:
            _skimage_ok = False
    return _skimage_ok


def available_models() -> list[dict]:
    """List models whose checkpoints exist on disk (drives the dropdown)."""
    out = []
    for name in ORDER:
        if name == "svm_hog":
            ok = all((MODELS_DIR / f).exists() for f in SVM_FILES) and _has_skimage()
        else:
            ok = (MODELS_DIR / DEEP_FILES[name]).exists()
        if ok:
            out.append({"id": name, **META[name]})
    return out


def load_model(name: str):
    """Lazy-load + cache a model. Returns a tagged tuple."""
    if name in _cache:
        return _cache[name]

    if name == "svm_hog":
        import joblib
        obj = ("svm",
               joblib.load(MODELS_DIR / "svm_model.pkl"),
               joblib.load(MODELS_DIR / "svm_scaler.pkl"),
               joblib.load(MODELS_DIR / "label_encoder.pkl"))
    else:
        model = build_model(name, dropout=CFG.dropout_head)
        state = torch.load(MODELS_DIR / DEEP_FILES[name], map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        obj = ("deep", model)

    _cache[name] = obj
    return obj


def _deep_probs(model, pil: Image.Image) -> np.ndarray:
    x = EVAL_TF(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return torch.softmax(model(x), dim=1)[0].cpu().numpy()


def _svm_probs(svm, scaler, le, pil: Image.Image) -> np.ndarray:
    import cv2
    from skimage.feature import hog
    gray = cv2.resize(np.array(pil.convert("L")), (128, 128))
    feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys")
    proba = svm.predict_proba(scaler.transform([feat]))[0]
    # Map the SVM's label-encoded columns onto the canonical CLASSES order.
    probs = np.zeros(len(CLASSES), dtype=float)
    for col, cls in enumerate(le.classes_):
        probs[CLASSES.index(cls)] = proba[col]
    return probs


# ──────────────────────────────────────────────────────────────────────────
# Grad-CAM — "where did the model look?"  (the visual explanation; deep models only)
# ──────────────────────────────────────────────────────────────────────────
def _last_conv(model) -> nn.Conv2d | None:
    """The last Conv2d layer — the standard Grad-CAM target: semantically rich, still spatial."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def _gradcam_overlay(model, pil: Image.Image) -> str | None:
    """Base64-PNG of the Grad-CAM heatmap blended over the (preprocessed) MRI.

    Forward + full-backward hooks on the last conv layer capture activations and
    gradients; we weight the activation maps by the pooled gradients, ReLU + normalize
    into a heatmap, then paint it (JET colormap) over the image the model actually saw.
    """
    layer = _last_conv(model)
    if layer is None:
        return None
    store: dict = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0].detach()))
    try:
        x = EVAL_TF(pil).unsqueeze(0).to(DEVICE).requires_grad_(True)
        out = model(x)
        idx = int(out.argmax(1).item())          # explain the PREDICTED class
        model.zero_grad(set_to_none=True)
        out[0, idx].backward()
        weights = store["g"].mean(dim=(0, 2, 3))                       # pool grads -> (C,)
        cam = torch.relu((weights[:, None, None] * store["a"][0]).sum(0))
        cam = cam.detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
    finally:
        h1.remove(); h2.remove()                  # never let hooks accumulate on a cached model

    base = np.array(_PRE(pil).resize((224, 224)).convert("RGB"))       # what the model saw
    cam = cv2.resize(cam, (224, 224))
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = np.uint8(base * 0.55 + heat * 0.45)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def predict(name: str, image_bytes: bytes) -> dict:
    """Run one image through the selected model. Returns prediction, scores, and Grad-CAM."""
    obj = load_model(name)
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if obj[0] == "deep":
        probs = _deep_probs(obj[1], pil)
        overlay = _gradcam_overlay(obj[1], pil)              # the "why" — where it looked
    else:
        probs = _svm_probs(obj[1], obj[2], obj[3], pil)
        overlay = None                                       # SVM is not a CNN — no Grad-CAM

    top = int(np.argmax(probs))
    scores = sorted(
        ({"label": CLASS_LABELS[i], "score": float(probs[i])} for i in range(len(CLASSES))),
        key=lambda d: d["score"], reverse=True,
    )
    return {
        "model": name,
        "model_label": META[name]["label"],
        "pred_label": CLASS_LABELS[top],
        "confidence": float(probs[top]),
        "scores": scores,
        "overlay": overlay,
        "explainable": overlay is not None,
        "device": DEVICE.type,
    }
