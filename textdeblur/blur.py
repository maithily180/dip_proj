"""Synthetic degradation: ``b = l * k + n``.

Used to build data with exact ground truth for the PSF.  The sharp image is
blurred *before* any cropping so that a crop taken from the result contains
content smeared in from outside its borders, exactly as in a real photograph.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .psf import motion_psf

__all__ = ["convolve", "add_noise", "degrade", "center_crop", "crop_window"]


def convolve(img: np.ndarray, psf: np.ndarray, mode: str = "reflect") -> np.ndarray:
    """Convolve a grayscale or colour image with ``psf`` (per channel)."""
    img = np.asarray(img, dtype=float)
    if img.ndim == 2:
        return ndimage.convolve(img, psf, mode=mode)
    return np.stack(
        [ndimage.convolve(img[..., c], psf, mode=mode) for c in range(img.shape[-1])], axis=-1
    )


def add_noise(
    img: np.ndarray, sigma: float, rng: np.random.Generator, quantize: bool = True
) -> np.ndarray:
    """Add zero-mean Gaussian noise (``sigma`` on a 0..1 scale), clip, and
    optionally quantise to 8 bits like a stored image."""
    out = np.asarray(img, dtype=float)
    if sigma > 0:
        out = out + rng.normal(0.0, sigma, out.shape)
    out = np.clip(out, 0.0, 1.0)
    if quantize:
        out = np.rint(out * 255.0) / 255.0
    return out


def degrade(
    sharp: np.ndarray,
    length: float,
    angle: float,
    noise_sigma: float = 0.0,
    seed: int | None = 0,
    quantize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Blur ``sharp`` with a linear motion PSF and add noise.

    Returns ``(blurred, psf)``.
    """
    psf = motion_psf(length, angle)
    rng = np.random.default_rng(seed)
    return add_noise(convolve(sharp, psf), noise_sigma, rng, quantize), psf


def center_crop(img: np.ndarray, height: int, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    top, left = (h - height) // 2, (w - width) // 2
    return img[top : top + height, left : left + width]


def crop_window(img: np.ndarray, cy: float, cx: float, height: int, width: int) -> np.ndarray:
    """Crop a ``height x width`` window centred on (cy, cx), shifted as needed
    to stay inside the image."""
    h, w = img.shape[:2]
    if h < height or w < width:
        raise ValueError(f"image {img.shape[:2]} smaller than crop {(height, width)}")
    top = int(round(cy - height / 2))
    left = int(round(cx - width / 2))
    top = min(max(top, 0), h - height)
    left = min(max(left, 0), w - width)
    return img[top : top + height, left : left + width]
