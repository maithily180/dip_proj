"""End-to-end blind deblurring: estimate -> gate -> deconvolve.

Three entry points:

* ``deblur``           - fully automatic (blind) restoration;
* ``deblur_with_psf``  - semi-blind restoration with a user-supplied
  (angle, length), e.g. marked by dragging along a visible streak;
* the ready-made configurations ``BASELINE_PIPELINE`` (classical recipe,
  always deconvolves) and ``TEXT_AWARE_PIPELINE`` (pre-processing + gate).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .deconv import richardson_lucy, wiener
from .estimate import BASELINE, TEXT_AWARE, BlurEstimate, EstimatorConfig, estimate_blur
from .gate import DEFAULT_GATE, GateConfig, GateDecision, decide
from .psf import motion_psf

__all__ = [
    "PipelineConfig", "DeblurResult", "deblur", "deblur_with_psf", "restore",
    "BASELINE_PIPELINE", "TEXT_AWARE_PIPELINE",
]


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration of the full pipeline.

    estimator : PSF estimator settings (including the pre-processing variant).
    gate      : gate thresholds, or ``None`` to always deconvolve.
    method    : ``"rl"`` (Richardson-Lucy, primary) or ``"wiener"``.
    num_iter  : Richardson-Lucy iterations.
    tv        : total-variation weight for Richardson-Lucy (0 = plain RL).
    nsr       : noise-to-signal ratio for Wiener.
    """

    estimator: EstimatorConfig = BASELINE
    gate: GateConfig | None = None
    method: str = "rl"
    num_iter: int = 30
    tv: float = 0.0
    nsr: float = 0.01

    def with_(self, **changes) -> "PipelineConfig":
        return replace(self, **changes)


@dataclass
class DeblurResult:
    """Output of ``deblur``."""

    restored: np.ndarray          # restored image, or the input if refused
    accepted: bool                # False when the gate withheld deconvolution
    reason: str                   # human-readable gate verdict
    estimate: BlurEstimate | None = field(repr=False, default=None)
    psf: np.ndarray | None = field(repr=False, default=None)
    decision: GateDecision | None = field(repr=False, default=None)


def restore(img: np.ndarray, psf: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Non-blind restoration of ``img`` with a known ``psf``."""
    if config.method == "rl":
        out = richardson_lucy(img, psf, num_iter=config.num_iter, tv=config.tv)
    elif config.method == "wiener":
        out = wiener(img, psf, nsr=config.nsr)
    else:
        raise ValueError(f"unknown deconvolution method '{config.method}'")
    return np.clip(out, 0.0, 1.0)


def deblur(img: np.ndarray, config: PipelineConfig | None = None) -> DeblurResult:
    """Blind restoration of a motion-blurred image.

    The PSF is estimated from ``img``; if a gate is configured and judges the
    estimate unreliable, the input is returned unchanged with
    ``accepted=False``.
    """
    cfg = config or TEXT_AWARE_PIPELINE
    img = np.asarray(img, dtype=float)
    est = estimate_blur(img, cfg.estimator)
    psf = est.psf()
    decision = decide(est, cfg.gate) if cfg.gate is not None else None
    if decision is not None and not decision.accept:
        return DeblurResult(img.copy(), False, decision.reason, est, psf, decision)
    reason = decision.reason if decision is not None else "gate disabled"
    return DeblurResult(restore(img, psf, cfg), True, reason, est, psf, decision)


def deblur_with_psf(
    img: np.ndarray, length: float, angle: float, config: PipelineConfig | None = None
) -> DeblurResult:
    """Semi-blind restoration: the user supplies the blur ``length`` (px) and
    ``angle`` (degrees CCW from +x) and only the non-blind step is run."""
    cfg = config or TEXT_AWARE_PIPELINE
    psf = motion_psf(length, angle)
    out = restore(np.asarray(img, dtype=float), psf, cfg)
    return DeblurResult(out, True, "user-supplied PSF", None, psf, None)


#: Classical recipe: plain luminance, Radon + cepstrum, always deconvolve.
BASELINE_PIPELINE = PipelineConfig(estimator=BASELINE, gate=None)

#: Text-aware pipeline: pre-processing stage plus confidence gate.
TEXT_AWARE_PIPELINE = PipelineConfig(estimator=TEXT_AWARE, gate=DEFAULT_GATE)
