"""Fourier-domain representations used by the PSF estimators.

A linear motion blur of length ``L`` multiplies the image spectrum by a
sinc-like transfer function whose zeros lie on parallel lines perpendicular to
the motion direction, spaced ``1/L`` cycles/pixel apart.  Everything in the
estimator is a way of detecting that stripe pattern.
"""
from __future__ import annotations

import numpy as np

__all__ = ["window2d", "prepare", "log_spectrum", "next_pow2", "mask_center"]


def next_pow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(int(n), 2))))


def window2d(shape: tuple[int, int], kind: str = "hann") -> np.ndarray:
    """Separable apodisation window that suppresses FFT boundary leakage."""
    h, w = shape
    if kind == "none":
        return np.ones(shape)
    if kind == "hann":
        wy, wx = np.hanning(h + 2)[1:-1], np.hanning(w + 2)[1:-1]
    elif kind == "tukey":
        from scipy.signal.windows import tukey

        wy, wx = tukey(h, 0.5), tukey(w, 0.5)
    else:
        raise ValueError(f"unknown window '{kind}'")
    return np.outer(wy, wx)


def prepare(img: np.ndarray, size: int | None = None, window: str = "hann") -> np.ndarray:
    """Mean-subtract, window, and zero-pad ``img`` to a square ``size x size``.

    Padding to a square makes the frequency grid isotropic, so a lag of ``q``
    samples in the cepstrum means ``q`` pixels in every direction even when the
    input crop is strongly rectangular (as plate crops are).
    """
    img = np.asarray(img, dtype=float)
    if img.ndim != 2:
        raise ValueError("prepare() expects a grayscale image")
    h, w = img.shape
    if size is None:
        size = next_pow2(max(h, w))
    if size < max(h, w):
        raise ValueError("size must be >= the largest image side")
    win = window2d((h, w), window)
    # subtract the window-weighted mean so the windowed signal has no DC term
    x = (img - float((img * win).sum() / win.sum())) * win
    out = np.zeros((size, size))
    top, left = (size - h) // 2, (size - w) // 2
    out[top : top + h, left : left + w] = x
    return out


def log_spectrum(
    img: np.ndarray, size: int | None = None, window: str = "hann", eps: float = 1e-3
) -> np.ndarray:
    """Centred log-magnitude spectrum ``log(|F| + eps * median|F|)``.

    The small floor keeps exact zeros finite without flattening the blur
    nulls, which are the signal we are after.
    """
    x = prepare(img, size=size, window=window)
    mag = np.abs(np.fft.fft2(x))
    floor = eps * float(np.median(mag)) + 1e-12
    return np.fft.fftshift(np.log(mag + floor))


def mask_center(spec: np.ndarray, radius: float, fill: float | None = None) -> np.ndarray:
    """Replace a disc around DC with ``fill`` (default: the spectrum median)."""
    n, m = spec.shape
    yy, xx = np.ogrid[:n, :m]
    r2 = (yy - n // 2) ** 2 + (xx - m // 2) ** 2
    out = spec.copy()
    out[r2 <= radius**2] = np.median(spec) if fill is None else fill
    return out
