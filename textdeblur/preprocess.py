"""Text-contrast pre-processing applied *before* PSF estimation.

The estimator only ever sees the pre-processed image; deconvolution is always
run on the original pixels.  The purpose is therefore not to make the image
look better but to strengthen the blur signature (the periodic nulls of the
PSF transfer function) relative to everything else in the log-spectrum.

Two families are provided:

* **Linear, shift-invariant** filters (``laplacian``, ``dog``, ``highpass``,
  ``unsharp``).  These commute with the blur, ``h*(k*l) = k*(h*l)``, so the
  blur nulls are untouched.  What they change is the *dynamic range* of the
  spectrum: text on a flat background has a steep low-frequency-dominated
  spectrum, and after windowing a small crop that energy leaks into and fills
  the nulls.  Pre-whitening removes the leak at its source.
* **Non-linear, local** operators (``clahe``, ``lcn``, ``homomorphic``,
  ``gradmag``).  These equalise stroke contrast across the crop so that
  unevenly lit or low-contrast strokes contribute as much as bright ones.
  They only approximately commute with the blur.

``text`` is the proposed stage: local contrast normalisation followed by
edge (Laplacian-of-Gaussian) emphasis, i.e. an edge-enhanced, locally
contrast-normalised representation.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from scipy import ndimage

from .data import to_gray

__all__ = ["PREPROCESSORS", "preprocess", "best_contrast_channel"]


def _gray(img: np.ndarray) -> np.ndarray:
    return to_gray(img)


def identity(img: np.ndarray) -> np.ndarray:
    """Plain luminance - the baseline input of the classical pipeline."""
    return _gray(img)


def clahe(img: np.ndarray, clip: float = 2.0, tiles: int = 4) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation."""
    g = np.rint(np.clip(_gray(img), 0, 1) * 255).astype(np.uint8)
    out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(g)
    return out.astype(float) / 255.0


def unsharp(img: np.ndarray, sigma: float = 2.0, amount: float = 1.5) -> np.ndarray:
    """Unsharp masking / high-boost filtering."""
    g = _gray(img)
    return g + amount * (g - ndimage.gaussian_filter(g, sigma))


def highpass(img: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Gaussian high-pass: removes illumination and flat background."""
    g = _gray(img)
    return g - ndimage.gaussian_filter(g, sigma)


def laplacian(img: np.ndarray) -> np.ndarray:
    """Discrete Laplacian (isotropic second derivative, linear)."""
    return ndimage.laplace(_gray(img), mode="reflect")


def dog(img: np.ndarray, s1: float = 0.6, s2: float = 2.0) -> np.ndarray:
    """Difference of Gaussians band-pass (stroke-scale edge emphasis)."""
    g = _gray(img)
    return ndimage.gaussian_filter(g, s1) - ndimage.gaussian_filter(g, s2)


def gradmag(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude (edge map)."""
    g = _gray(img)
    return np.hypot(ndimage.sobel(g, axis=0), ndimage.sobel(g, axis=1))


def lcn(img: np.ndarray, sigma: float = 8.0, eps: float = 0.02) -> np.ndarray:
    """Local contrast normalisation: subtract the local mean and divide by the
    local standard deviation (plus ``eps`` so flat regions stay near zero)."""
    g = _gray(img)
    mu = ndimage.gaussian_filter(g, sigma)
    var = ndimage.gaussian_filter((g - mu) ** 2, sigma)
    return (g - mu) / (np.sqrt(np.maximum(var, 0)) + eps)


def homomorphic(img: np.ndarray, sigma: float = 10.0, low: float = 0.3, high: float = 1.5) -> np.ndarray:
    """Homomorphic filtering: attenuate illumination, boost reflectance."""
    g = np.log1p(np.clip(_gray(img), 0, None) * 255.0)
    lp = ndimage.gaussian_filter(g, sigma)
    out = low * lp + high * (g - lp)
    return np.expm1(out) / 255.0


def best_contrast_channel(img: np.ndarray) -> np.ndarray:
    """Colour channel with the largest robust contrast (inter-decile range).

    Plates are saturated colours with near-white or near-black characters, so
    one channel usually separates strokes from background far better than
    luminance does.  Grayscale input passes through unchanged.
    """
    img = np.asarray(img, dtype=float)
    if img.ndim == 2:
        return img
    spread = [np.subtract(*np.percentile(img[..., c], [90, 10])) for c in range(img.shape[-1])]
    return img[..., int(np.argmax(spread))]


def text(img: np.ndarray, sigma: float = 8.0, eps: float = 0.02) -> np.ndarray:
    """Proposed text-contrast stage: best-contrast channel -> local contrast
    normalisation -> Laplacian edge emphasis."""
    g = lcn(best_contrast_channel(img), sigma=sigma, eps=eps)
    return ndimage.laplace(g, mode="reflect")


PREPROCESSORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "gray": identity,
    "clahe": clahe,
    "unsharp": unsharp,
    "highpass": highpass,
    "laplacian": laplacian,
    "dog": dog,
    "gradmag": gradmag,
    "lcn": lcn,
    "homomorphic": homomorphic,
    "channel": best_contrast_channel,
    "text": text,
}


def preprocess(img: np.ndarray, method: str = "gray") -> np.ndarray:
    """Apply a named pre-processing variant and return a 2-D float image."""
    try:
        fn = PREPROCESSORS[method]
    except KeyError as exc:
        raise ValueError(f"unknown pre-processing '{method}'; choose from {sorted(PREPROCESSORS)}") from exc
    return np.asarray(fn(img), dtype=float)
