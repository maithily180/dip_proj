"""The synthetic sweep engine shared by experiments 2-4.

One *trial* = one sharp source blurred with one (angle, length) at one noise
level.  For every trial, every requested estimator variant is run on the same
blurred crop, so variants are compared on identical data.  For variants listed
in ``restore_for``, the image is also restored with the estimated PSF so that the
outcome can be judged in image space (PSNR/SSIM against the sharp crop, and
against the true-PSF restoration as the ceiling); for variants in
``agreement_for`` the split-half agreement used by the gate is recorded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    ANGLE_TOL, ANGLES, LENGTH_TOL, LENGTHS, MAX_LENGTH, NOISES, Source, make_trial, pmap, sources,
)
from textdeblur.deconv import richardson_lucy
from textdeblur.estimate import EstimatorConfig, estimate_blur
from textdeblur.gate import consistency
from textdeblur.metrics import angle_error, psnr, ssim
from textdeblur.psf import motion_psf
from textdeblur.radon_angle import estimate_two_stage

#: Richardson-Lucy settings used for every restoration in the sweeps
RL_ITERS = 100
RL_STOP = 1.0
BORDER = 8


def restore(blurred: np.ndarray, psf: np.ndarray) -> np.ndarray:
    return np.clip(richardson_lucy(blurred, psf, num_iter=RL_ITERS, stop=RL_STOP), 0, 1)


def _trial(job):
    src, length, angle, sigma, configs, restore_for, full_frame, agreement_for = job
    sharp, blurred = make_trial(src, length, angle, sigma, full_frame=full_frame)
    cls = "natural_full" if full_frame else src.cls
    base = dict(cls=cls, image=src.name, split=src.split, length=length, angle=angle, sigma=sigma)

    quality = {}
    if restore_for:
        quality["psnr_blur"] = psnr(blurred, sharp, BORDER)
        quality["ssim_blur"] = ssim(blurred, sharp, BORDER)
        r_true = restore(blurred, motion_psf(length, angle))
        quality["psnr_true"] = psnr(r_true, sharp, BORDER)
        quality["ssim_true"] = ssim(r_true, sharp, BORDER)

    rows = []
    for name, cfg in configs.items():
        if name == "two_stage":
            est = estimate_two_stage(blurred, cfg.with_(max_length=MAX_LENGTH))
        else:
            est = estimate_blur(blurred, cfg.with_(max_length=MAX_LENGTH))
        a_err = angle_error(est.angle, angle)
        l_err = abs(est.length - length)
        row = dict(base, variant=name, angle_est=est.angle, length_est=est.length,
                   angle_err=a_err, length_err=l_err,
                   success=bool(a_err <= ANGLE_TOL and l_err <= LENGTH_TOL),
                   confidence=est.confidence, peak_ratio=est.peak_ratio, depth=est.depth,
                   angular_z=est.angular_z, cutoff=est.cutoff)
        if name in agreement_for:
            row["agreement"] = consistency(blurred, est)
        if name in restore_for:
            r_est = restore(blurred, est.psf())
            row.update(quality, psnr_est=psnr(r_est, sharp, BORDER), ssim_est=ssim(r_est, sharp, BORDER))
        rows.append(row)
    return rows


def run_sweep(
    configs: dict[str, EstimatorConfig],
    classes: tuple[str, ...] = ("natural", "text", "plate"),
    split: str | None = "test",
    restore_for: tuple[str, ...] = (),
    full_frame_natural: bool = False,
    desc: str = "sweep",
    agreement_for: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Run every variant in ``configs`` over the full angle x length x noise
    grid for the chosen classes and split.  Returns one row per (trial, variant)."""
    srcs: list[Source] = [s for s in sources() if s.cls in classes and (split is None or s.split == split)]
    ag = tuple(agreement_for)
    jobs = [(s, L, a, n, configs, tuple(restore_for), False, ag)
            for s in srcs for L in LENGTHS for a in ANGLES for n in NOISES]
    if full_frame_natural:
        jobs += [(s, L, a, n, configs, tuple(restore_for), True, ag)
                 for s in sources() if s.cls == "natural" and (split is None or s.split == split)
                 for L in LENGTHS for a in ANGLES for n in NOISES]
    rows = [r for chunk in pmap(_trial, jobs, desc=desc, chunksize=2) for r in chunk]
    return pd.DataFrame(rows)


def success_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Success rate (%) pivoted by ``by`` with one column per class/variant."""
    return (100 * df.pivot_table(index=by, columns=["cls", "variant"], values="success")).round(1)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per class and variant: success rate with a 95% CI over source images,
    plus median angle / length errors."""
    out = []
    for (cls, variant), g in df.groupby(["cls", "variant"], sort=False):
        per_img = g.groupby("image").success.mean()
        se = per_img.std(ddof=1) / np.sqrt(len(per_img)) if len(per_img) > 1 else 0.0
        out.append(dict(
            cls=cls, variant=variant, n_trials=len(g), n_images=len(per_img),
            success_pct=100 * g.success.mean(), ci95_pct=100 * 1.96 * se,
            median_angle_err=g.angle_err.median(), mean_angle_err=g.angle_err.mean(),
            median_length_err=g.length_err.median(), mean_length_err=g.length_err.mean(),
        ))
    return pd.DataFrame(out)
