"""Joint (angle, length) estimation of a linear motion PSF.

Pipeline
--------
1. (optional) text-contrast pre-processing of the blurred image;
2. windowed log-magnitude spectrum on a square, isotropic frequency grid;
3. (optional) spectral clean-up: clipping of JPEG block-grid lines and an
   SNR-adaptive band limit;
4. Radon transform of the log-spectrum over 180 directions;
5. 1-D Fourier transform of every projection, giving the radial cepstrum
   profile for that direction (projection-slice theorem);
6. the deepest negative cepstral dip over all (direction, lag) pairs gives the
   blur angle and length;
7. the depth of that dip relative to the background of the whole search
   region is the confidence score used by the gate.

The angle found in step 6 is "Radon-based" (it is the projection direction
whose profile carries the strongest periodic null pattern) and the length is
"cepstrum-based" (the lag of the dip along that direction); doing both in one
search is what makes the classical recipe robust.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from scipy import ndimage

from .preprocess import preprocess
from .psf import motion_psf
from .spectrum import log_spectrum, next_pow2

__all__ = [
    "EstimatorConfig", "BlurEstimate", "estimate_blur", "radon_transform", "BASELINE", "TEXT_AWARE",
]


@dataclass(frozen=True)
class EstimatorConfig:
    """Settings of the PSF estimator.

    preprocess      : name of the pre-processing variant (see ``preprocess``).
    window          : spatial apodisation window (``tukey`` or ``hann``).
    min_length      : smallest blur length searched (px).
    max_length      : largest blur length searched; ``None`` = 30% of the
                      longer image side.
    angle_step      : coarse angular resolution of the Radon sweep (degrees).
    refine          : re-sweep +-1.5 deg around the best angle at 0.25 deg.
    cutoff          : radial band limit as a fraction of Nyquist, or ``"auto"``
                      to stop where the spectrum reaches its noise floor.
    notch_grid      : clip the 8x8 block-grid lines that JPEG compression adds
                      to the spectrum.
    angular_detrend : subtract, at every lag, the median over directions; this
                      removes isotropic rings so only *directional* dips count.
    min_size        : minimum FFT size.
    """

    preprocess: str = "gray"
    window: str = "tukey"
    min_length: float = 4.0
    max_length: float | None = None
    angle_step: float = 1.0
    refine: bool = True
    cutoff: float | str = 1.0
    notch_grid: bool = False
    angular_detrend: bool = False
    min_size: int = 256

    def with_(self, **changes) -> "EstimatorConfig":
        return replace(self, **changes)


@dataclass
class BlurEstimate:
    """Result of ``estimate_blur``."""

    angle: float                 # degrees, CCW from +x, in [0, 180)
    length: float                # pixels (sub-pixel)
    depth: float                 # depth of the winning cepstral dip (>0)
    confidence: float            # dip depth in robust sigmas above background
    peak_ratio: float            # winning dip / best competing dip elsewhere
    angles: np.ndarray = field(repr=False, default=None)      # coarse sweep angles
    lags: np.ndarray = field(repr=False, default=None)        # lag axis of ``profiles``
    profiles: np.ndarray = field(repr=False, default=None)    # (n_angles, n_lags)
    spectrum: np.ndarray = field(repr=False, default=None)    # processed log-spectrum
    cutoff: float = 1.0
    config: EstimatorConfig = field(repr=False, default=None)

    def psf(self) -> np.ndarray:
        """The estimated line kernel."""
        return motion_psf(self.length, self.angle)

    @property
    def profile(self) -> np.ndarray:
        """Cepstral profile along the estimated direction."""
        i = int(np.argmin(np.abs(((self.angles - self.angle + 90) % 180) - 90)))
        return self.profiles[i]


# --------------------------------------------------------------------------
# spectral clean-up
# --------------------------------------------------------------------------


def clip_block_grid(spec: np.ndarray, block: int = 8, half_width: int = 1) -> np.ndarray:
    """Suppress the bright lines JPEG's 8x8 block grid adds to a spectrum.

    Blocking is periodic with period ``block`` pixels, so it puts energy on the
    lines ``u = k/block`` and ``v = k/block`` cycles/pixel.  Each such line is
    clipped *from above* to the level of its neighbours.  Blur nulls are dark,
    not bright, so a genuine null that happens to fall on the grid survives.
    """
    n = spec.shape[0]
    out = spec.copy()
    c = n // 2
    off = half_width + 2
    for k in range(1, block // 2 + 1):
        for sign in (-1, 1):
            pos = c + sign * k * n // block
            if not (off <= pos < n - off):
                continue
            for d in range(-half_width, half_width + 1):
                p = pos + d
                ref_c = np.minimum(out[:, pos - off], out[:, pos + off])
                out[:, p] = np.minimum(out[:, p], ref_c)
                ref_r = np.minimum(out[pos - off, :], out[pos + off, :])
                out[p, :] = np.minimum(out[p, :], ref_r)
    return out


def auto_cutoff(spec: np.ndarray, margin: float = 0.35, lo: float = 0.35) -> float:
    """Radius (fraction of Nyquist) at which the log-spectrum reaches its
    high-frequency floor.

    Beyond that radius the spectrum is noise or compression residue and carries
    no blur nulls; including it only adds variance to the cepstral profiles.
    """
    n = spec.shape[0]
    yy, xx = np.indices(spec.shape)
    r = np.hypot(yy - n // 2, xx - n // 2)
    nb = n // 2
    idx = np.minimum(r.astype(int), nb)
    sums = np.bincount(idx.ravel(), spec.ravel(), minlength=nb + 1)[:nb]
    cnts = np.bincount(idx.ravel(), minlength=nb + 1)[:nb]
    radial = ndimage.uniform_filter1d(sums / np.maximum(cnts, 1), max(3, nb // 32))
    floor = np.median(radial[int(0.9 * nb):])
    above = np.nonzero(radial > floor + margin)[0]
    if len(above) == 0:
        return lo
    return float(np.clip((above[-1] + 1) / nb, lo, 1.0))


def _disc_taper(n: int, cutoff: float) -> np.ndarray:
    yy, xx = np.ogrid[:n, :n]
    r = np.hypot(yy - n // 2, xx - n // 2) / (n / 2.0 * cutoff)
    return np.where(r < 1, 0.5 * (1 + np.cos(np.pi * np.clip(r, 0, 1))), 0.0)


# --------------------------------------------------------------------------
# Radon-cepstrum profiles
# --------------------------------------------------------------------------


def radon_transform(img: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Radon transform of a square image by rotate-and-sum.

    Column ``i`` of the result is the projection of ``img`` onto the axis at
    ``angles[i]`` degrees: the image is rotated so that the integration lines
    become vertical and then summed down the columns.  This is numerically
    identical to ``skimage.transform.radon(img, angles, circle=True)`` (the
    test suite checks it) but an order of magnitude faster, which matters
    because the estimator calls it for 180 directions per image.
    """
    n = img.shape[0]
    src = np.ascontiguousarray(img, dtype=np.float32)
    out = np.empty((n, len(angles)), dtype=np.float64)
    for i, a in enumerate(np.asarray(angles, dtype=float)):
        m = cv2.getRotationMatrix2D((n // 2, n // 2), -float(a), 1.0)
        out[:, i] = cv2.warpAffine(src, m, (n, n), flags=cv2.INTER_LINEAR).sum(axis=0, dtype=np.float64)
    return out


def _radon_profiles(spec: np.ndarray, angles: np.ndarray, n_lags: int) -> np.ndarray:
    n = spec.shape[0]
    sino = radon_transform(spec, angles)
    prof = np.real(np.fft.fft(np.fft.ifftshift(sino, axes=0), axis=0)) / float(n * n)
    return prof[:n_lags].T


def _parabolic_min(y: np.ndarray, i: int) -> float:
    """Sub-sample position of a minimum from its two neighbours."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    denom = a - 2 * b + c
    if denom <= 1e-12:
        return float(i)
    return float(i + np.clip(0.5 * (a - c) / denom, -0.5, 0.5))


def estimate_blur(img: np.ndarray, config: EstimatorConfig | None = None) -> BlurEstimate:
    """Estimate the linear motion PSF of a blurred image.

    ``img`` may be grayscale or RGB in [0, 1].  Returns a ``BlurEstimate``.
    """
    cfg = config or BASELINE
    g = preprocess(img, cfg.preprocess)
    h, w = g.shape
    n = max(cfg.min_size, next_pow2(max(h, w)))

    spec = log_spectrum(g, size=n, window=cfg.window)
    if cfg.notch_grid:
        spec = clip_block_grid(spec)
    cutoff = auto_cutoff(spec) if cfg.cutoff == "auto" else float(cfg.cutoff)
    taper = _disc_taper(n, cutoff)
    inside = taper > 0
    spec_w = (spec - np.median(spec[inside])) * taper

    max_len = cfg.max_length if cfg.max_length is not None else 0.3 * max(h, w)
    lo = int(np.ceil(cfg.min_length))
    hi = int(min(np.floor(max_len), n // 2 - 2))
    if hi <= lo:
        raise ValueError("image too small for the requested blur-length range")

    angles = np.arange(0.0, 180.0, cfg.angle_step)
    prof = _radon_profiles(spec_w, angles, hi + 2)
    if cfg.angular_detrend:
        prof = prof - np.median(prof, axis=0, keepdims=True)

    region = prof[:, lo : hi + 1]
    ia, il = np.unravel_index(int(np.argmin(region)), region.shape)
    angle = float(angles[ia])
    best_profile = prof[ia]

    if cfg.refine:
        fine = (angle + np.arange(-1.5, 1.51, 0.25)) % 180.0
        fprof = _radon_profiles(spec_w, fine, hi + 2)
        if cfg.angular_detrend:
            fprof = fprof - np.median(prof, axis=0, keepdims=True)
        fregion = fprof[:, max(lo, il + lo - 1) : min(hi, il + lo + 1) + 1]
        fa, _ = np.unravel_index(int(np.argmin(fregion)), fregion.shape)
        angle = float(fine[fa])
        best_profile = fprof[fa]

    q = int(np.argmin(best_profile[lo : hi + 1])) + lo
    if abs(q - (il + lo)) > 1:          # refinement must not jump to another dip
        q = il + lo
    length = _parabolic_min(best_profile, q)
    depth = float(-best_profile[q])

    # confidence: how far the winning dip sticks out of the background of
    # the whole (direction, lag) search region, in robust standard deviations
    med = float(np.median(region))
    mad = float(np.median(np.abs(region - med))) * 1.4826 + 1e-12
    confidence = (med - float(region[ia, il])) / mad

    # best competing dip outside the neighbourhood of the winner
    dth = np.abs(((angles - angles[ia] + 90) % 180) - 90)[:, None]
    dq = np.abs(np.arange(region.shape[1]) - il)[None, :]
    rival = np.where((dth <= 8) & (dq <= 2), np.inf, region)
    rival_depth = med - float(rival.min())
    peak_ratio = float((med - float(region[ia, il])) / max(rival_depth, 1e-12))

    return BlurEstimate(
        angle=angle % 180.0, length=float(length), depth=depth, confidence=float(confidence),
        peak_ratio=peak_ratio, angles=angles, lags=np.arange(prof.shape[1]), profiles=prof,
        spectrum=spec_w, cutoff=cutoff, config=cfg,
    )


#: The classical pipeline as it stands in the literature: plain luminance in,
#: Radon + cepstrum out.
BASELINE = EstimatorConfig()

#: The text-aware estimator (pre-processing stage switched on).  The concrete
#: settings are chosen on the development split; see ``experiments/``.
TEXT_AWARE = EstimatorConfig(preprocess="text")
