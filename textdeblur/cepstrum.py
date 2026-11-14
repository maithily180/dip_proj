"""Cepstral analysis of the blur signature.

The cepstrum is ``C = IFFT(log|FFT(b)|)``.  Because ``b = l * k`` becomes
``log|B| = log|L| + log|K|`` in the log-spectrum, the blur contributes an
*additive* term to the cepstrum.  For a linear motion blur of length ``L`` that
term is a sharp negative spike at distance ``L`` from the origin along the blur
direction (in 1-D, ``log|K|`` has cepstrum ``-1/(2m)`` at lags ``mL``).

Two equivalent ways of getting radial cepstrum profiles are implemented:

* ``polar_cepstrum``   - sample the 2-D cepstrum along rays through the origin;
* ``radon_cepstrum``   - Radon-transform the log-spectrum and take the 1-D
  Fourier transform of every projection.

By the projection-slice theorem these are the same quantity: the 1-D spectrum
of a Radon projection of ``log|B|`` at angle ``theta`` *is* the slice of the
2-D cepstrum along ``theta``.  This is what ties the "Radon on the spectrum"
angle estimator and the "cepstrum along that direction" length estimator into
one combined search.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage
from skimage.transform import radon

from .spectrum import log_spectrum

__all__ = ["cepstrum2d", "polar_cepstrum", "radon_cepstrum", "radial_weight"]


def radial_weight(n: int, cutoff: float = 1.0, kind: str = "hann") -> np.ndarray:
    """Isotropic taper on an ``n x n`` centred spectrum.

    ``cutoff`` is the radius (as a fraction of Nyquist) where the weight
    reaches zero.  Tapering removes the square-support corners (so the Radon
    transform sees a disc) and de-emphasises the highest frequencies, where
    sensor noise rather than blur nulls dominates.
    """
    yy, xx = np.ogrid[:n, :n]
    r = np.hypot(yy - n // 2, xx - n // 2) / (n / 2.0 * cutoff)
    if kind == "hann":
        w = 0.5 * (1 + np.cos(np.pi * np.clip(r, 0, 1)))
    elif kind == "disc":
        w = (r <= 1).astype(float)
    else:
        raise ValueError(kind)
    w[r >= 1] = 0.0
    return w


def weighted_log_spectrum(
    img: np.ndarray,
    size: int | None = None,
    window: str = "hann",
    cutoff: float = 1.0,
    taper: str = "hann",
) -> np.ndarray:
    """Log-spectrum, median-centred and radially tapered (zero outside disc)."""
    spec = log_spectrum(img, size=size, window=window)
    w = radial_weight(spec.shape[0], cutoff, taper)
    inside = w > 0
    return (spec - np.median(spec[inside])) * w


def cepstrum2d(
    img: np.ndarray,
    size: int | None = None,
    window: str = "hann",
    cutoff: float = 1.0,
    taper: str = "hann",
) -> np.ndarray:
    """Centred 2-D cepstrum of ``img`` (origin at ``[n//2, n//2]``)."""
    spec = weighted_log_spectrum(img, size, window, cutoff, taper)
    ceps = np.real(np.fft.ifft2(np.fft.ifftshift(spec)))
    return np.fft.fftshift(ceps)


def polar_cepstrum(ceps: np.ndarray, angles: np.ndarray, max_radius: int) -> np.ndarray:
    """Sample a centred cepstrum on rays: result[i, q] = C(q at angle angles[i]).

    Angles are in degrees, counter-clockwise from +x with y pointing up.
    """
    n = ceps.shape[0]
    c = n // 2
    q = np.arange(max_radius + 1)
    th = np.deg2rad(np.asarray(angles, dtype=float))[:, None]
    xs = c + q[None, :] * np.cos(th)
    ys = c - q[None, :] * np.sin(th)
    return ndimage.map_coordinates(ceps, [ys, xs], order=1, mode="nearest")


def radon_cepstrum(spec: np.ndarray, angles: np.ndarray, max_radius: int) -> np.ndarray:
    """Radial cepstrum profiles from the Radon transform of a log-spectrum.

    ``spec`` must be a centred ``n x n`` (weighted) log-spectrum that is zero
    outside the inscribed disc.  Returns an array of shape
    ``(len(angles), max_radius + 1)`` on the same scale as ``cepstrum2d``.
    """
    n = spec.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sino = radon(spec, theta=np.asarray(angles, dtype=float), circle=True)
    # sino[:, i] is the projection onto the axis at angles[i]; its centre
    # sample (n // 2) corresponds to the spectrum origin.
    proj = np.fft.ifftshift(sino, axes=0)
    prof = np.real(np.fft.fft(proj, axis=0)) / float(n * n)
    return prof[: max_radius + 1].T
