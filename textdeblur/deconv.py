"""Non-blind deconvolution: Richardson-Lucy (primary) and Wiener (comparison).

Boundary handling
-----------------
FFT-based deconvolution assumes the image is periodic.  A photograph is not,
and the mismatch shows up as dark bands and ringing that start at the borders.
Both restorers here work on a padded canvas:

* Richardson-Lucy treats the padding as *unobserved*.  The data-fidelity term
  is masked to the real pixels, and the estimate is free to take whatever
  values in the margin best explain the blurred data (the "unknown boundary"
  form of RL).  No pixel values are invented.
* Wiener is a single linear filter and cannot ignore pixels, so the canvas is
  filled by edge replication and cross-faded to make it periodic (the classic
  edge-taper), then cropped back.
"""
from __future__ import annotations

import numpy as np

from .psf import pad_psf_to

__all__ = ["richardson_lucy", "wiener", "deconvolve", "blur_fft"]


def _fast_len(n: int) -> int:
    from scipy.fft import next_fast_len

    return int(next_fast_len(int(n), real=True))


def _per_channel(fn, img: np.ndarray, *args, **kwargs) -> np.ndarray:
    img = np.asarray(img, dtype=float)
    if img.ndim == 2:
        return fn(img, *args, **kwargs)
    return np.stack([fn(img[..., c], *args, **kwargs) for c in range(img.shape[-1])], axis=-1)


def blur_fft(img: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Circular convolution of a 2-D image with a centred PSF (via FFT)."""
    otf = np.fft.rfft2(pad_psf_to(psf, img.shape))
    return np.fft.irfft2(np.fft.rfft2(img) * otf, s=img.shape)


# --------------------------------------------------------------------------
# Richardson-Lucy
# --------------------------------------------------------------------------


def _rl_gray(
    b: np.ndarray, psf: np.ndarray, num_iter: int, pad: int | None, tv: float, floor: float
) -> np.ndarray:
    h, w = b.shape
    if pad is None:
        pad = int(max(psf.shape))
    H, W = _fast_len(h + 2 * pad), _fast_len(w + 2 * pad)
    shape = (H, W)

    data = np.zeros(shape)
    mask = np.zeros(shape)
    data[pad : pad + h, pad : pad + w] = np.maximum(b, 0.0)
    mask[pad : pad + h, pad : pad + w] = 1.0

    otf = np.fft.rfft2(pad_psf_to(psf, shape))
    otf_c = np.conj(otf)
    conv = lambda x, f: np.fft.irfft2(np.fft.rfft2(x) * f, s=shape)

    # sensitivity of each latent pixel to the observed region; latent pixels
    # the data never sees (outer margin) are left at their initial value
    norm = np.maximum(conv(mask, otf_c), 1e-3)

    est = np.pad(np.maximum(b, floor), ((pad, H - h - pad), (pad, W - w - pad)), mode="edge")
    for _ in range(int(num_iter)):
        reblur = conv(est, otf)
        ratio = mask * data / np.maximum(reblur, floor)
        update = conv(ratio, otf_c) / norm
        if tv > 0:
            gy, gx = np.gradient(est)
            mag = np.sqrt(gx**2 + gy**2 + 1e-8)
            div = np.gradient(gy / mag, axis=0) + np.gradient(gx / mag, axis=1)
            update = update / np.maximum(1.0 - tv * div, 0.5)
        est = np.maximum(est * update, 0.0)
    return est[pad : pad + h, pad : pad + w]


def richardson_lucy(
    img: np.ndarray,
    psf: np.ndarray,
    num_iter: int = 30,
    pad: int | None = None,
    tv: float = 0.0,
    floor: float = 1e-4,
) -> np.ndarray:
    """Richardson-Lucy deconvolution with unknown-boundary handling.

    Parameters
    ----------
    img      : blurred image in [0, 1], grayscale or colour.
    psf      : blur kernel (non-negative, sums to 1).
    num_iter : number of multiplicative updates.  More iterations sharpen
               further but amplify noise and ringing.
    pad      : margin (pixels) of unobserved canvas around the image;
               defaults to the PSF size.
    tv       : optional total-variation regularisation weight (0 = plain RL).
    """
    return _per_channel(_rl_gray, img, psf, num_iter, pad, tv, floor)


# --------------------------------------------------------------------------
# Wiener
# --------------------------------------------------------------------------


def _periodic_canvas(b: np.ndarray, pad: int) -> tuple[np.ndarray, int]:
    """Edge-replicate ``b`` onto a larger canvas and cross-fade the margin so
    the canvas tiles without a seam."""
    h, w = b.shape
    H, W = _fast_len(h + 2 * pad), _fast_len(w + 2 * pad)
    canvas = np.pad(b, ((pad, H - h - pad), (pad, W - w - pad)), mode="edge")

    def ramp(n_total: int, n_img: int) -> np.ndarray:
        wgt = np.ones(n_total)
        lead, trail = pad, n_total - n_img - pad
        wgt[:lead] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(lead) + 0.5) / lead) if lead else 1
        if trail:
            wgt[n_total - trail :] = 0.5 + 0.5 * np.cos(np.pi * (np.arange(trail) + 0.5) / trail)
        return wgt

    wgt = np.outer(ramp(H, h), ramp(W, w))
    mean = float(b.mean())
    return wgt * canvas + (1 - wgt) * mean, pad


def _wiener_gray(b: np.ndarray, psf: np.ndarray, nsr: float, pad: int | None) -> np.ndarray:
    h, w = b.shape
    if pad is None:
        pad = int(2 * max(psf.shape))
    canvas, pad = _periodic_canvas(b, pad)
    otf = np.fft.rfft2(pad_psf_to(psf, canvas.shape))
    filt = np.conj(otf) / (np.abs(otf) ** 2 + nsr)
    out = np.fft.irfft2(np.fft.rfft2(canvas) * filt, s=canvas.shape)
    return out[pad : pad + h, pad : pad + w]


def wiener(img: np.ndarray, psf: np.ndarray, nsr: float = 0.01, pad: int | None = None) -> np.ndarray:
    """Wiener deconvolution ``conj(K) / (|K|^2 + nsr)`` with edge tapering.

    ``nsr`` is the (constant) noise-to-signal power ratio; larger values give a
    smoother, less ringing-prone result.
    """
    return _per_channel(_wiener_gray, img, psf, nsr, pad)


def deconvolve(img: np.ndarray, psf: np.ndarray, method: str = "rl", **kwargs) -> np.ndarray:
    """Dispatch to ``richardson_lucy`` (``'rl'``) or ``wiener`` (``'wiener'``)."""
    if method == "rl":
        return richardson_lucy(img, psf, **kwargs)
    if method == "wiener":
        return wiener(img, psf, **kwargs)
    raise ValueError(f"unknown deconvolution method '{method}'")
