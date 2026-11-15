"""Experiment 3 - does pre-processing strengthen the blur signature?

Every candidate is inserted in front of the *same* estimator and run on the
*same* trials as experiment 2 (proposal objective 3).  The candidates are

* the classical contrast/edge operators the proposal set out to test
  (CLAHE, unsharp masking, Gaussian high-pass, Laplacian, DoG, gradient
  magnitude, local contrast normalisation, homomorphic filtering, and the
  edge-enhanced locally-normalised combination ``text``);
* the stage the analysis led to instead: restricting the spectrum to the band
  where the image has signal above its noise floor - a single SNR-adaptive
  limit (``band``), fixed limits (``band0.5``, ``band0.3``), combinations with
  contrast enhancement, and the final multi-band search (``text_aware``).

Outputs: results/csv/03_preprocess.csv.gz, 03_summary.csv, results/figures/03_*.png
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import CLASS_LABELS, CSV, FIG, NOISES, setup_matplotlib
from sweep import run_sweep, summary
from textdeblur.estimate import BASELINE, TEXT_AWARE

VARIANTS = {
    "gray": BASELINE,
    "clahe": BASELINE.with_(preprocess="clahe"),
    "unsharp": BASELINE.with_(preprocess="unsharp"),
    "homomorphic": BASELINE.with_(preprocess="homomorphic"),
    "lcn": BASELINE.with_(preprocess="lcn"),
    "highpass": BASELINE.with_(preprocess="highpass"),
    "dog": BASELINE.with_(preprocess="dog"),
    "laplacian": BASELINE.with_(preprocess="laplacian"),
    "gradmag": BASELINE.with_(preprocess="gradmag"),
    "text": BASELINE.with_(preprocess="text"),
    "band": BASELINE.with_(cutoff="auto"),
    "band0.5": BASELINE.with_(cutoff=0.5),
    "band0.3": BASELINE.with_(cutoff=0.3),
    "band+clahe": BASELINE.with_(cutoff="auto", preprocess="clahe"),
    "band+lcn": BASELINE.with_(cutoff="auto", preprocess="lcn"),
    "text_aware": TEXT_AWARE,
}
LABELS = {
    "gray": "plain grayscale (baseline)", "clahe": "CLAHE", "unsharp": "unsharp mask",
    "homomorphic": "homomorphic", "lcn": "local contrast norm.", "highpass": "Gaussian high-pass",
    "dog": "difference of Gaussians", "laplacian": "Laplacian", "gradmag": "gradient magnitude",
    "text": "LCN + Laplacian (proposed)", "band": "adaptive band limit", "band0.5": "band limit 0.5", "band0.3": "band limit 0.3", "band+clahe": "band + CLAHE",
    "band+lcn": "band + LCN", "text_aware": "multi-band search (final)",
}
ORDER = ["natural", "text", "plate"]


def main():
    if "--plot-only" in sys.argv:                     # redraw from a previous run
        df = pd.read_csv(CSV / "03_preprocess.csv.gz")
    else:
        df = run_sweep(VARIANTS, split="test", restore_for=("gray", "text_aware"),
                       agreement_for=("gray", "text_aware"), desc="pre-processing sweep")
        df.to_csv(CSV / "03_preprocess.csv.gz", index=False)
    summ = summary(df)
    summ.to_csv(CSV / "03_summary.csv", index=False)

    pd.set_option("display.width", 220)
    tab = (100 * df.pivot_table(index="variant", columns="cls", values="success")).loc[list(VARIANTS), ORDER]
    tab["mean"] = tab.mean(axis=1)
    print("\n== Success rate (%) by pre-processing variant, test split =="); print(tab.round(1).to_string())
    print("\n== Change vs. plain grayscale (percentage points) ==")
    print((tab - tab.loc["gray"]).round(1).to_string())

    print("\n== Success (%) by noise level ==")
    bn = 100 * df.pivot_table(index="variant", columns=["cls", "sigma"], values="success")
    print(bn.loc[list(VARIANTS), ORDER].round(0).to_string())
    print("\n== Success (%) by blur length ==")
    bl = 100 * df.pivot_table(index="variant", columns=["cls", "length"], values="success")
    print(bl.loc[list(VARIANTS), ORDER].round(0).to_string())

    q = df[df.variant.isin(["gray", "text_aware"])]
    print("\n== Restoration with the estimated PSF (mean PSNR dB; harmful = worse than blurred) ==")
    g = q.assign(harm=q.psnr_est < q.psnr_blur).groupby(["cls", "variant"])
    print(pd.concat([g[["psnr_blur", "psnr_est", "psnr_true"]].mean(), 100 * g.harm.mean()], axis=1).round(2).to_string())

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    names = list(VARIANTS)
    x = np.arange(len(names)); w = 0.27
    colors = {"natural": "#8c8c8c", "text": "#dd8452", "plate": "#2a6fb0"}
    for k, cls in enumerate(ORDER):
        s = summ[summ.cls == cls].set_index("variant").loc[names]
        ax.bar(x + (k - 1) * w, s.success_pct, w, yerr=s.ci95_pct, capsize=2, color=colors[cls],
               label=CLASS_LABELS[cls], error_kw=dict(lw=0.8))
    for cls in ORDER:
        ax.axhline(tab.loc["gray", cls], color=colors[cls], lw=0.8, ls=":")
    ax.set_xticks(x, [LABELS[n] for n in names], rotation=32, ha="right")
    ax.set_ylabel("PSF estimation success (%)"); ax.set_ylim(0, 105)
    ax.set_title("Effect of pre-processing on PSF estimation (dotted lines: plain grayscale)", pad=24)
    ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 0.99))
    fig.savefig(FIG / "03_preprocess_success.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), sharey=True)
    show = ["gray", "clahe", "lcn", "highpass", "text", "band", "text_aware"]
    for ax, cls in zip(axes, ORDER):
        for n in show:
            m = 100 * df[(df.cls == cls) & (df.variant == n)].groupby("sigma").success.mean()
            style = dict(lw=2.4, color="k") if n == "gray" else dict(lw=2.4, color="#c0392b") if n == "text_aware" else dict(lw=1.2)
            ax.plot(range(len(NOISES)), m.values, "o-", ms=3.5, label=LABELS[n], **style)
        ax.set_xticks(range(len(NOISES)), [f"{s:g}" for s in NOISES])
        ax.set_title(CLASS_LABELS[cls], fontsize=9.5); ax.set_xlabel("noise sigma")
    axes[0].set_ylabel("success (%)"); axes[2].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("Pre-processing variants under increasing noise", y=1.02)
    fig.savefig(FIG / "03_preprocess_noise.png"); plt.close(fig)
    print(f"\nwrote results/csv/03_preprocess.csv.gz and figures 03_*.png")


if __name__ == "__main__":
    main()
