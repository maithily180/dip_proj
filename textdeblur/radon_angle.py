"""The textbook two-stage rule, kept as a reference point.

Stage 1 picks the blur angle from the Radon transform of the log-spectrum
alone: the projection taken across the null stripes oscillates the most, so
the angle with the largest projection variance is taken as the blur
direction.  Stage 2 then reads the blur length off the cepstrum along that
single direction.

This is the recipe as usually stated.  Its weakness is that stage 1 is
decided before any cepstral evidence is consulted, so the image's own
dominant edge directions (which also produce high-variance projections)
compete with the blur stripes.  ``estimate.estimate_blur`` replaces it with a
joint search; this module exists so that the difference can be measured.
"""
from __future__ import annotations

import warnings

import numpy as np
from skimage.transform import radon

from .estimate import BlurEstimate, EstimatorConfig, _disc_taper, _parabolic_min, _radon_profiles
from .preprocess import preprocess
from .spectrum import log_spectrum, mask_center, next_pow2

__all__ = ["radon_angle", "estimate_two_stage"]


def radon_angle(spec: np.ndarray, angle_step: float = 1.0) -> tuple[float, np.ndarray, np.ndarray]:
    """Blur angle from projection variance of a centred log-spectrum.

    Returns ``(angle, angles, variances)``.
    """
    angles = np.arange(0.0, 180.0, angle_step)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sino = radon(spec, theta=angles, circle=True)
    n = spec.shape[0]
    core = sino[n // 8 : n - n // 8]              # ignore the short chords near the rim
    var = core.var(axis=0)
    return float(angles[int(np.argmax(var))]), angles, var


def estimate_two_stage(img: np.ndarray, config: EstimatorConfig | None = None) -> BlurEstimate:
    """Radon-variance angle followed by a cepstral length along that angle."""
    cfg = config or EstimatorConfig()
    g = preprocess(img, cfg.preprocess)
    h, w = g.shape
    n = max(cfg.min_size, next_pow2(max(h, w)))
    spec = log_spectrum(g, size=n, window=cfg.window)
    taper = _disc_taper(n, 1.0)
    spec_w = (spec - np.median(spec[taper > 0])) * taper

    angle, angles, _ = radon_angle(mask_center(spec_w, 3, 0.0), cfg.angle_step)

    max_len = cfg.max_length if cfg.max_length is not None else 0.3 * max(h, w)
    lo = int(np.ceil(cfg.min_length))
    hi = int(min(np.floor(max_len), n // 2 - 2))
    prof = _radon_profiles(spec_w, np.array([angle]), hi + 2)[0]
    q = int(np.argmin(prof[lo : hi + 1])) + lo
    seg = prof[lo : hi + 1]
    med = float(np.median(seg))
    mad = float(np.median(np.abs(seg - med))) * 1.4826 + 1e-12
    return BlurEstimate(
        angle=angle, length=_parabolic_min(prof, q), depth=float(-prof[q]),
        confidence=(med - float(prof[q])) / mad, peak_ratio=1.0,
        angles=np.array([angle]), lags=np.arange(len(prof)), profiles=prof[None, :],
        spectrum=spec_w, cutoff=1.0, config=cfg,
    )
