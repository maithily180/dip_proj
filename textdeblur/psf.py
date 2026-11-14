"""Linear uniform motion point-spread functions.

Under the linear, spatially uniform motion model the blur kernel is fully
described by two scalars: the blur angle ``theta`` and the blur length ``L``.

Conventions used throughout the project
---------------------------------------
* ``angle`` is in degrees, measured counter-clockwise from the +x axis as the
  image is displayed (so 0 deg is horizontal motion, 90 deg is vertical).  A
  line kernel is symmetric, so angles are only meaningful modulo 180.
* ``length`` is the extent of the motion in pixels.  A horizontal kernel of
  integer length ``L`` is exactly ``L`` equal taps of ``1/L``, so its transfer
  function has zeros at multiples of ``1/L`` cycles/pixel and its cepstrum has
  a negative spike at lag ``L``.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = ["motion_psf", "psf_support", "pad_psf_to", "angle_diff"]


def angle_diff(a: float, b: float) -> float:
    """Absolute difference between two line orientations (degrees, mod 180)."""
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def psf_support(length: float) -> int:
    """Odd kernel side length that comfortably contains a blur of ``length``."""
    size = int(np.ceil(length)) + 3
    return size + 1 if size % 2 == 0 else size


@lru_cache(maxsize=4096)
def _motion_psf_cached(length: float, angle: float, size: int) -> np.ndarray:
    c = size // 2
    if length <= 1.0:
        k = np.zeros((size, size))
        k[c, c] = 1.0
        return k

    theta = np.deg2rad(angle)
    # Area sampling: each pixel's weight is the length of the motion segment
    # that falls inside that pixel's square.  Integrate numerically by
    # sampling the segment densely and binning every sample to its pixel.
    n_t = int(max(512, np.ceil(length * 256)))
    t = (np.arange(n_t) + 0.5) / n_t * length - length / 2.0
    x = c + t * np.cos(theta)
    y = c - t * np.sin(theta)                 # rows grow downward
    # round half away from the centre so the kernel stays point-symmetric
    xi = (np.sign(x - c) * np.floor(np.abs(x - c) + 0.5)).astype(int) + c
    yi = (np.sign(y - c) * np.floor(np.abs(y - c) + 0.5)).astype(int) + c
    ok = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
    k = np.zeros((size, size))
    np.add.at(k, (yi[ok], xi[ok]), 1.0)
    k /= k.sum()
    k.setflags(write=False)
    return k


def motion_psf(length: float, angle: float, size: int | None = None) -> np.ndarray:
    """Anti-aliased line kernel for linear uniform motion.

    Parameters
    ----------
    length : blur extent in pixels (values <= 1 give a delta kernel).
    angle  : blur direction in degrees, counter-clockwise from +x.
    size   : optional odd side length of the returned square array.

    Returns
    -------
    A non-negative ``size x size`` array that sums to 1.
    """
    length = float(length)
    if size is None:
        size = psf_support(length)
    if size % 2 == 0:
        raise ValueError("PSF size must be odd")
    k = _motion_psf_cached(round(length, 4), round(float(angle) % 180.0, 4), int(size))
    return k.copy()


def pad_psf_to(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Embed ``psf`` in a zero array of ``shape`` with its centre at index (0, 0).

    The result is suitable for ``np.fft.fft2`` so that multiplication in the
    frequency domain equals circular convolution with the centred kernel.
    """
    out = np.zeros(shape, dtype=float)
    kh, kw = psf.shape
    out[:kh, :kw] = psf
    return np.roll(out, (-(kh // 2), -(kw // 2)), axis=(0, 1))
