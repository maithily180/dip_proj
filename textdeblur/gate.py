"""Confidence gating: decide whether an estimated PSF is safe to deconvolve with.

Deconvolution is an inverse filter.  Given the wrong kernel it does not merely
fail to sharpen, it amplifies the wrong frequencies and produces ringing and
ghost edges that leave the image *worse* than the blurred input.  The gate
turns that silent failure into an explicit "not confident - image returned
unchanged".

Four tests are available; each is switched off by a threshold of 0.

* ``confidence``  - depth of the winning cepstral dip against the background
  of the whole (direction, lag) search region, in robust standard deviations;
* ``angular_z``   - depth of the winning dip against the *other directions at
  the same lag*.  Isotropic structure (soft or band-limited content, defocus,
  noise texture) is deep in every direction and scores near zero, a genuine
  linear-motion dip exists in one direction only;
* ``peak_ratio``  - winning dip over the best competing dip elsewhere, which
  rejects spectra with several comparable candidates;
* ``agreement``   - a uniform blur is the same everywhere, so the estimate is
  repeated on the left/right and top/bottom halves of the image and the
  number of halves (0-4) that reproduce it is counted.  Defocus, non-uniform
  shake and compression artefacts rarely give the same answer twice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimate import BlurEstimate, EstimatorConfig, estimate_blur
from .psf import angle_diff

__all__ = [
    "GateConfig", "GateDecision", "decide", "roc_curve", "auc", "calibrate_threshold",
    "confusion", "consistency", "DEFAULT_GATE", "STRICT_GATE",
]


@dataclass(frozen=True)
class GateConfig:
    """Thresholds of the confidence gate."""

    min_confidence: float = 0.0
    min_angular_z: float = 0.0
    min_peak_ratio: float = 0.0
    min_agreement: int = 0

    @property
    def needs_agreement(self) -> bool:
        return self.min_agreement > 0


@dataclass(frozen=True)
class GateDecision:
    accept: bool
    reason: str
    confidence: float
    peak_ratio: float
    angular_z: float = 0.0
    agreement: int | None = None


def consistency(img: np.ndarray, estimate: BlurEstimate, config: EstimatorConfig | None = None,
                angle_tol: float = 10.0, length_tol: float = 0.15) -> int:
    """Number of image halves (left, right, top, bottom) whose own estimate
    reproduces ``estimate`` (angle within ``angle_tol`` degrees and length
    within max(2 px, ``length_tol`` x length))."""
    cfg = config or estimate.config
    h, w = img.shape[:2]
    full_max = cfg.max_length if cfg.max_length is not None else 0.3 * max(h, w)
    count = 0
    for sub in (img[:, : w // 2], img[:, w // 2 :], img[: h // 2], img[h // 2 :]):
        try:
            e = estimate_blur(sub, cfg.with_(max_length=min(full_max, 0.45 * max(sub.shape[:2]))))
        except ValueError:
            continue
        if angle_diff(e.angle, estimate.angle) <= angle_tol and                 abs(e.length - estimate.length) <= max(2.0, length_tol * estimate.length):
            count += 1
    return count


def decide(estimate: BlurEstimate, gate: GateConfig, agreement: int | None = None) -> GateDecision:
    """Accept or refuse an estimate.  ``agreement`` (from ``consistency``) is
    required only when ``gate.min_agreement > 0``."""
    c, r, z = estimate.confidence, estimate.peak_ratio, estimate.angular_z
    d = lambda ok, why: GateDecision(ok, why, c, r, z, agreement)
    if c < gate.min_confidence:
        return d(False, f"cepstral dip too weak ({c:.1f} < {gate.min_confidence:.1f} sigma)")
    if z < gate.min_angular_z:
        return d(False, f"dip not directional ({z:.1f} < {gate.min_angular_z:.1f} sigma across angles)")
    if r < gate.min_peak_ratio:
        return d(False, f"ambiguous: rival dip nearly as deep (ratio {r:.2f} < {gate.min_peak_ratio:.2f})")
    if gate.min_agreement > 0:
        if agreement is None:
            raise ValueError("this gate needs the split-half agreement; pass consistency(...)")
        if agreement < gate.min_agreement:
            return d(False, f"not uniform: only {agreement}/4 image halves agree (< {gate.min_agreement})")
    extra = f", {agreement}/4 halves agree" if agreement is not None else ""
    return d(True, f"confident ({c:.1f} sigma, directional {z:.1f} sigma{extra})")


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


#: Gate calibrated on the development data (synthetic dev split, LPBlur and
#: CCPD-Blur dev subsets) by ``experiments/04_gate_calibration.py``: the
#: thresholds with the best mean net share (net gain let through / total
#: stakes) over the three sets.
DEFAULT_GATE = GateConfig(min_confidence=3.0, min_agreement=2)

#: The strictest operating point of the same calibration: the most helpful
#: restorations kept while harming at most 5% of the images of every dev set.
STRICT_GATE = GateConfig(min_agreement=3)
