"""Experiment 2 - where does the classical pipeline work, and where does it fail?

A controlled sweep over blur angle x blur length x noise level on uniform
synthetic blur (proposal objective 2), comparing

* natural, texture-rich images (Kodak) at full frame - how the classical
  recipe is normally evaluated;
* the same natural images cut to a plate-sized window;
* printed text and license plates in that same window.

Separating *content* from *region size* matters: a license plate is not only
low-texture, it is also a small region, and the two effects are easy to
confuse.  The textbook two-stage rule (Radon-variance angle, then cepstral
length) and an unrestricted joint (direction x lag) search are run alongside
the baseline estimator (joint refinement within +-3 deg of the Radon angle) as
reference points.

Outputs: results/csv/02_sweep.csv, 02_summary.csv, results/figures/02_*.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import ANGLES, CLASS_LABELS, CSV, FIG, LENGTHS, NOISES, setup_matplotlib
from sweep import run_sweep, summary
from textdeblur.estimate import BASELINE

ORDER = ["natural_full", "natural", "text", "plate"]


def main():
    df = run_sweep({"baseline": BASELINE, "two_stage": BASELINE, "joint_search": BASELINE.with_(angle_window=None)},
                   split="test",
                   restore_for=("baseline",), full_frame_natural=True, desc="characterisation sweep")
    df.to_csv(CSV / "02_sweep.csv", index=False)
    summ = summary(df)
    summ.to_csv(CSV / "02_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n== Success rate (angle within 5 deg AND length within 2 px), test split ==")
    print(summ.round(2).to_string(index=False))

    base = df[df.variant == "baseline"]
    print("\n== Baseline estimator: success (%) by noise level ==")
    print((100 * base.pivot_table(index="cls", columns="sigma", values="success")).loc[ORDER].round(1).to_string())
    print("\n== Baseline estimator: success (%) by blur length ==")
    print((100 * base.pivot_table(index="cls", columns="length", values="success")).loc[ORDER].round(1).to_string())
    print("\n== Baseline estimator: success (%) by blur angle ==")
    print((100 * base.pivot_table(index="cls", columns="angle", values="success")).loc[ORDER].round(1).to_string())

    print("\n== Restoration quality with the estimated PSF (baseline estimator), mean dB ==")
    q = base.groupby("cls")[["psnr_blur", "psnr_est", "psnr_true"]].mean().loc[ORDER]
    q["harmful_pct"] = 100 * base.assign(h=base.psnr_est < base.psnr_blur).groupby("cls").h.mean().loc[ORDER]
    print(q.round(2).to_string())

    plt = setup_matplotlib()

    # --- figure: success-rate bars, two estimators x four conditions
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    x = np.arange(len(ORDER)); w = 0.27
    for k, (variant, label, color) in enumerate([("joint_search", "unrestricted joint search", "#d9d9d9"),
                                                 ("two_stage", "textbook two-stage rule", "#a0a0a0"),
                                                 ("baseline", "Radon angle + joint refinement (baseline)", "#2a6fb0")]):
        s = summ[summ.variant == variant].set_index("cls").loc[ORDER]
        ax.bar(x + (k - 1) * w, s.success_pct, w, yerr=s.ci95_pct, capsize=2.5, label=label, color=color)
        for xi, v in zip(x + (k - 1) * w, s.success_pct):
            ax.text(xi, v + 2.5, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, [CLASS_LABELS[c].replace(", ", "\n") for c in ORDER])
    ax.set_ylabel("PSF estimation success (%)"); ax.set_ylim(0, 112)
    ax.set_title("Classical PSF estimation by image class (all angles, lengths, noise levels)")
    ax.legend(frameon=False, loc="upper center", fontsize=8, ncol=3)
    fig.savefig(FIG / "02_success_by_class.png"); plt.close(fig)

    # --- figure: heatmaps length x noise per class
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, cls in zip(axes, ORDER):
        m = 100 * base[base.cls == cls].pivot_table(index="length", columns="sigma", values="success")
        im = ax.imshow(m.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(NOISES)), [f"{s:g}" for s in NOISES])
        ax.set_yticks(range(len(LENGTHS)), LENGTHS)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m.values[i, j]:.0f}", ha="center", va="center", fontsize=9)
        ax.set_title(CLASS_LABELS[cls], fontsize=9.5); ax.set_xlabel("noise sigma")
    axes[0].set_ylabel("blur length (px)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="success (%)")
    fig.suptitle("Where the baseline fails: success rate by blur length and noise", y=1.02)
    fig.savefig(FIG / "02_heatmap_length_noise.png"); plt.close(fig)

    # --- figure: heatmaps angle x noise per class (mean angle error)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), sharey=True)
    for ax, cls in zip(axes, ORDER):
        m = base[base.cls == cls].pivot_table(index="angle", columns="sigma", values="angle_err")
        im = ax.imshow(m.values, cmap="RdYlGn_r", vmin=0, vmax=30, aspect="auto")
        ax.set_xticks(range(len(NOISES)), [f"{s:g}" for s in NOISES])
        ax.set_yticks(range(len(ANGLES)), [f"{a:g}" for a in ANGLES])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m.values[i, j]:.0f}", ha="center", va="center", fontsize=8.5)
        ax.set_title(CLASS_LABELS[cls], fontsize=9.5); ax.set_xlabel("noise sigma")
    axes[0].set_ylabel("true blur angle (deg)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="mean angle error (deg)")
    fig.suptitle("Mean angle error by blur angle and noise", y=1.02)
    fig.savefig(FIG / "02_heatmap_angle_noise.png"); plt.close(fig)
    print(f"\nwrote {CSV / '02_sweep.csv'} and figures 02_*.png")


if __name__ == "__main__":
    main()
