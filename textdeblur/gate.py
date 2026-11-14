"""Confidence gating: decide whether an estimated PSF is safe to deconvolve with.

Deconvolution is an inverse filter.  Given the wrong kernel it does not merely
fail to sharpen, it amplifies the wrong frequencies and produces ringing and
ghost edges that leave the image *worse* than the blurred input.  The gate
turns that silent failure into an explicit "not confident - image returned
unchanged".

The decision variable is the strength of the detected cepstral dip relative to
the background of the search region (``BlurEstimate.confidence``, in robust
standard deviations).  A second, optional test requires the winning dip to
beat the best competing dip elsewhere by a margin (``peak_ratio``), which
rejects ambiguous spectra with several comparable candidates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimate import BlurEstimate

__all__ = [
    "GateConfig", "GateDecision", "decide", "roc_curve", "auc", "calibrate_threshold",
    "confusion", "DEFAULT_GATE",
]


@dataclass(frozen=True)
class GateConfig:
    """Thresholds of the confidence gate."""

    min_confidence: float = 0.0
    min_peak_ratio: float = 0.0


@dataclass(frozen=True)
class GateDecision:
    accept: bool
    reason: str
    confidence: float
    peak_ratio: float


def decide(estimate: BlurEstimate, gate: GateConfig) -> GateDecision:
    """Accept or refuse an estimate."""
    c, r = estimate.confidence, estimate.peak_ratio
    if c < gate.min_confidence:
        return GateDecision(False, f"cepstral dip too weak ({c:.1f} < {gate.min_confidence:.1f} sigma)", c, r)
    if r < gate.min_peak_ratio:
        return GateDecision(False, f"ambiguous: rival dip nearly as deep (ratio {r:.2f} < {gate.min_peak_ratio:.2f})", c, r)
    return GateDecision(True, f"confident ({c:.1f} sigma, ratio {r:.2f})", c, r)


# --------------------------------------------------------------------------
# calibration utilities
# --------------------------------------------------------------------------


def roc_curve(scores: np.ndarray, good: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC of "score >= t  ->  accept" for detecting good estimates.

    Returns ``(false_accept_rate, true_accept_rate, thresholds)`` with
    thresholds in decreasing order.
    """
    scores = np.asarray(scores, dtype=float)
    good = np.asarray(good, dtype=bool)
    order = np.argsort(-scores, kind="stable")
    s, g = scores[order], good[order]
    tp = np.cumsum(g)
    fp = np.cumsum(~g)
    distinct = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tpr = np.r_[0.0, tp[distinct] / max(int(g.sum()), 1)]
    fpr = np.r_[0.0, fp[distinct] / max(int((~g).sum()), 1)]
    thr = np.r_[np.inf, s[distinct]]
    return fpr, tpr, thr


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under an ROC curve (trapezoidal rule)."""
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def calibrate_threshold(scores: np.ndarray, good: np.ndarray, max_false_accept: float = 0.05) -> float:
    """Smallest threshold whose false-accept rate (bad estimates let through)
    does not exceed ``max_false_accept``.  This maximises the number of good
    restorations kept subject to a cap on harmful ones."""
    fpr, _, thr = roc_curve(scores, good)
    ok = np.nonzero(fpr <= max_false_accept)[0]
    return float(thr[ok[-1]]) if len(ok) else float("inf")


def confusion(accept: np.ndarray, good: np.ndarray) -> dict[str, int]:
    """2x2 gate confusion matrix.

    ``good``   - the restoration (had it been run) is better than the input.
    ``accept`` - the gate let the restoration through.
    """
    accept = np.asarray(accept, dtype=bool)
    good = np.asarray(good, dtype=bool)
    return {
        "correct_restore": int(np.sum(accept & good)),      # proceed, and it helps
        "harmful_restore": int(np.sum(accept & ~good)),     # proceed, but it hurts
        "missed_restore": int(np.sum(~accept & good)),      # refuse a good one
        "avoided_harm": int(np.sum(~accept & ~good)),       # refuse a bad one
    }


#: Gate thresholds calibrated on the synthetic development split
#: (``experiments/04_gate_calibration.py``).
DEFAULT_GATE = GateConfig(min_confidence=0.0, min_peak_ratio=0.0)
