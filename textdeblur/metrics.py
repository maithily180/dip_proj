"""Evaluation metrics.

Tier 1 - PSF accuracy:      ``angle_error``, ``length_error``, ``kernel_similarity``
Tier 2 - restoration:       ``psnr``, ``ssim`` (both with a border crop)
Tier 3 - downstream/no-ref: ``laplacian_variance``, ``tenengrad``, text metrics
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal
from skimage.metrics import structural_similarity

from .data import to_gray
from .psf import angle_diff

__all__ = [
    "angle_error", "length_error", "psnr", "ssim", "crop_border", "kernel_similarity",
    "laplacian_variance", "tenengrad", "edit_distance", "char_accuracy",
]


def angle_error(estimated: float, true: float) -> float:
    """Orientation error in degrees (0..90), accounting for the 180 deg wrap."""
    return angle_diff(estimated, true)


def length_error(estimated: float, true: float) -> float:
    return abs(float(estimated) - float(true))


def crop_border(img: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return img
    return img[border:-border, border:-border]


def psnr(test: np.ndarray, ref: np.ndarray, border: int = 0) -> float:
    """Peak signal-to-noise ratio in dB for images on a [0, 1] scale."""
    a = crop_border(np.clip(np.asarray(test, dtype=float), 0, 1), border)
    b = crop_border(np.clip(np.asarray(ref, dtype=float), 0, 1), border)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def ssim(test: np.ndarray, ref: np.ndarray, border: int = 0) -> float:
    """Structural similarity (computed on luminance)."""
    a = crop_border(np.clip(to_gray(test), 0, 1), border)
    b = crop_border(np.clip(to_gray(ref), 0, 1), border)
    return float(structural_similarity(a, b, data_range=1.0))


def kernel_similarity(k1: np.ndarray, k2: np.ndarray) -> float:
    """Maximum normalised cross-correlation between two kernels over all
    shifts (Hu & Yang, 2012).  1 means identical up to translation."""
    a = np.asarray(k1, dtype=float)
    b = np.asarray(k2, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(signal.correlate2d(a, b, mode="full").max() / denom)


def laplacian_variance(img: np.ndarray) -> float:
    """Variance of the Laplacian - a no-reference sharpness score."""
    return float(ndimage.laplace(to_gray(img)).var())


def tenengrad(img: np.ndarray) -> float:
    """Mean squared Sobel gradient magnitude - a no-reference sharpness score."""
    g = to_gray(img)
    return float(np.mean(ndimage.sobel(g, axis=0) ** 2 + ndimage.sobel(g, axis=1) ** 2))


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(predicted: str, truth: str) -> float:
    """Character-level accuracy ``1 - edit_distance / len(truth)``, floored at 0."""
    if not truth:
        return 0.0
    return max(0.0, 1.0 - edit_distance(predicted, truth) / len(truth))
