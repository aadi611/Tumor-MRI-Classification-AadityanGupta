"""
preprocessing.py
================
Image-level preprocessing that runs *before* tensor conversion / augmentation.

Two domain-specific steps that materially help brain-MRI classification:

1. Brain-region cropping
   Raw MRI slices contain a large black border around the skull. Cropping to the
   tight bounding box of the brain removes uninformative background, so a larger
   fraction of the input resolution is spent on the actual anatomy. This is a
   well-known trick for the Br35H / Sartaj brain-tumor datasets.

2. CLAHE (Contrast-Limited Adaptive Histogram Equalisation)
   MRI intensity ranges vary a lot between scanners/sequences. CLAHE locally
   normalises contrast so tumour texture becomes more consistent across images
   without blowing out noise (the "clip limit" caps amplification).

Both are implemented as callables that take and return a PIL.Image, so they slot
directly into a torchvision `transforms.Compose` via `transforms.Lambda`.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def crop_brain_region(img: np.ndarray, add_pixels: int = 0) -> np.ndarray:
    """Crop a grayscale/colour MRI to the tight bounding box of the brain.

    Strategy: threshold -> morphological clean-up -> largest external contour ->
    extreme points -> bounding box. Falls back to the original image if no
    contour is found (e.g. an almost-empty slice).

    Parameters
    ----------
    img : np.ndarray
        H×W (grayscale) or H×W×3 (RGB) uint8 array.
    add_pixels : int
        Optional padding kept around the detected brain box.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold adapts to each image's intensity distribution.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Close small holes then erode/dilate to drop specks.
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img  # nothing detected — return unchanged

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Guard against degenerate boxes (tiny artefacts).
    if w < 10 or h < 10:
        return img

    p = add_pixels
    H, W = gray.shape[:2]
    y0, y1 = max(0, y - p), min(H, y + h + p)
    x0, x1 = max(0, x - p), min(W, x + w + p)
    return img[y0:y1, x0:x1]


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0,
                grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE. For RGB input it equalises the luminance (L) channel only,
    preserving colour relationships; for grayscale it equalises directly."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return clahe.apply(img)


class MRIPreprocess:
    """PIL-in / PIL-out preprocessing transform (crop + CLAHE).

    Usable inside torchvision pipelines:
        transforms.Compose([MRIPreprocess(cfg), transforms.Resize(...), ...])
    """

    def __init__(self, use_crop: bool = True, use_clahe: bool = True,
                 clahe_clip_limit: float = 2.0, clahe_grid: tuple[int, int] = (8, 8)):
        self.use_crop = use_crop
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid = clahe_grid

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("RGB"))           # ensure 3-channel uint8
        if self.use_crop:
            arr = crop_brain_region(arr)
        if self.use_clahe:
            arr = apply_clahe(arr, self.clahe_clip_limit, self.clahe_grid)
        return Image.fromarray(arr)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(crop={self.use_crop}, "
                f"clahe={self.use_clahe})")
