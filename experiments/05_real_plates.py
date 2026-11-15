"""Experiment 5 - the full pipeline on real blurred license plates (objective 5).

Two public datasets (see ``real_data.py``):

* **LPBlur**   - real motion blur from moving vehicles, with paired sharp
  exposures (OCR of the sharp image is the reference text; PSNR/SSIM are
  also available);
* **CCPD-Blur** - real blurred plates in street photographs, annotated text,
  no sharp counterpart.  Most of this blur is *not* uniform linear motion
  (defocus, mixed and non-uniform shake), which makes it the natural test of
  whether the gate refuses what the model cannot explain.

Systems compared on the same held-out images (none of them was used for any
design decision):

=====================  =====================================================
``blurred``            the input, unchanged
``baseline``           classical recipe: grayscale Radon-cepstrum, RL, always
``text_aware``         final estimator (multi-band search), RL, always
``text_aware_gated``   ... with the confidence gate (refused -> input kept)
``baseline_gated``     the gate applied to the classical estimator
``wiener``             text-aware estimate restored with Wiener (ablation)
=====================  =====================================================

OCR (recognition-only, on rectified plates) is the read-out; it is never used
inside the pipeline.  Every mean carries a 95% bootstrap interval.

Outputs: results/csv/05_real_<dataset>.csv, 05_summary.csv, results/figures/05_*.png
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import CSV, FIG, pmap, setup_matplotlib
from real_data import ccpd_items, lpblur_items, load_item
from textdeblur.deconv import richardson_lucy, wiener
from textdeblur.estimate import BASELINE, TEXT_AWARE, estimate_blur
from textdeblur.gate import DEFAULT_GATE, consistency, decide
from textdeblur.metrics import char_accuracy, laplacian_variance, psnr, ssim

RL_ITERS = 30
WIENER_NSR = 0.03
SYSTEMS = ["blurred", "baseline", "baseline_gated", "text_aware", "text_aware_gated", "wiener"]
LABELS = {
    "blurred": "blurred input", "baseline": "classical baseline", "baseline_gated": "baseline + gate",
    "text_aware": "text-aware (no gate)", "text_aware_gated": "text-aware + gate", "wiener": "text-aware, Wiener",
}


def rl(img, psf):
    return np.clip(richardson_lucy(img, psf, num_iter=RL_ITERS), 0, 1)


def run(item):
    it = load_item(item)
    b = it.blurred
    out = {"blurred": b}
    row = dict(dataset=item.dataset, name=item.name, truth=it.truth)
    for name, cfg in (("baseline", BASELINE), ("text_aware", TEXT_AWARE)):
        est = estimate_blur(b, cfg)
        agree = consistency(b, est, cfg) if DEFAULT_GATE.needs_agreement else None
        dec = decide(est, DEFAULT_GATE, agree)
        row.update({f"{name}_angle": est.angle, f"{name}_length": est.length, f"{name}_conf": est.confidence,
                    f"{name}_angz": est.angular_z, f"{name}_ratio": est.peak_ratio, f"{name}_cutoff": est.cutoff,
                    f"{name}_agree": agree, f"{name}_accept": dec.accept})
        out[name] = rl(b, est.psf())
        out[f"{name}_gated"] = out[name] if dec.accept else b
        if name == "text_aware":
            out["wiener"] = np.clip(wiener(b, est.psf(), nsr=WIENER_NSR), 0, 1)
    for s in SYSTEMS:
        text = it.reader(out[s])
        row[f"ocr_{s}"] = text
        row[f"acc_{s}"] = char_accuracy(text, it.truth)
        row[f"exact_{s}"] = text == it.truth
        row[f"lapvar_{s}"] = laplacian_variance(out[s])
        if it.sharp is not None:
            row[f"psnr_{s}"] = psnr(out[s], it.sharp, 4)
            row[f"ssim_{s}"] = ssim(out[s], it.sharp, 4)
    return row


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x))).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    before = df.acc_blurred.values
    for s in SYSTEMS:
        a = df[f"acc_{s}"].values
        lo, hi = boot_ci(a)
        r = dict(dataset=dataset, system=s, n=len(df), char_acc=a.mean(), ci_lo=lo, ci_hi=hi,
                 exact=df[f"exact_{s}"].mean(), improved=np.mean(a > before), worse=np.mean(a < before))
        if f"psnr_{s}" in df:
            r.update(psnr=df[f"psnr_{s}"].mean(), ssim=df[f"ssim_{s}"].mean())
        if s.endswith("_gated"):
            r["accepted"] = df[f"{s[:-6]}_accept"].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def gate_table(df: pd.DataFrame, name: str = "text_aware") -> dict:
    """Gate confusion on real data: 'good' = restoration raises OCR accuracy."""
    acc, before, accept = df[f"acc_{name}"].values, df.acc_blurred.values, df[f"{name}_accept"].values
    good, bad = acc > before, acc < before
    return dict(accepted=int(accept.sum()), refused=int((~accept).sum()),
                accepted_helped=int((accept & good).sum()), accepted_hurt=int((accept & bad).sum()),
                refused_would_help=int((~accept & good).sum()), refused_would_hurt=int((~accept & bad).sum()),
                neutral=int((~good & ~bad).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="test images per dataset")
    args = ap.parse_args()

    summaries, frames = [], {}
    for dataset, items in (("lpblur", lpblur_items(args.n, "test")), ("ccpd", ccpd_items(args.n, "test"))):
        rows = pmap(run, items, chunksize=2, desc=f"real plates [{dataset}]")
        df = pd.DataFrame(rows)
        if dataset == "lpblur":         # reference text must be readable from the sharp image
            n0 = len(df)
            df = df[df.truth.str.len() == 6].reset_index(drop=True)
            print(f"LPBlur: {len(df)} of {n0} pairs have a 6-character reading of the sharp image")
        df.to_csv(CSV / f"05_real_{dataset}.csv", index=False)
        frames[dataset] = df
        summaries.append(summarise(df, dataset))
    summ = pd.concat(summaries, ignore_index=True)
    summ.to_csv(CSV / "05_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n== OCR character accuracy on held-out real plates ==")
    print(summ.round(3).to_string(index=False))
    for dataset, df in frames.items():
        print(f"\n== Gate on {dataset} (good = OCR accuracy rises, hurt = falls) ==")
        for name in ("baseline", "text_aware"):
            print(f"  {name:10s}", gate_table(df, name))
        print("  median estimated length: baseline %.1f, text-aware %.1f px; mean band limit %.2f"
              % (df.baseline_length.median(), df.text_aware_length.median(), df.text_aware_cutoff.mean()))

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    colors = {"blurred": "#8c8c8c", "baseline": "#b0b0b0", "baseline_gated": "#d9d9d9",
              "text_aware": "#7fa7d0", "text_aware_gated": "#2a6fb0", "wiener": "#dd8452"}
    for ax, (dataset, title) in zip(axes, [("lpblur", "LPBlur (real motion blur)"),
                                            ("ccpd", "CCPD-Blur (mostly non-linear blur)")]):
        s = summ[summ.dataset == dataset].set_index("system").loc[SYSTEMS]
        x = np.arange(len(SYSTEMS))
        ax.bar(x, 100 * s.char_acc, color=[colors[k] for k in SYSTEMS],
               yerr=[100 * (s.char_acc - s.ci_lo), 100 * (s.ci_hi - s.char_acc)], capsize=3)
        ax.axhline(100 * s.loc["blurred", "char_acc"], color="k", lw=0.8, ls=":")
        ax.set_xticks(x, [LABELS[k] for k in SYSTEMS], rotation=28, ha="right")
        ax.set_title(f"{title}, n={int(s.n.iloc[0])}"); ax.set_ylim(0, 100)
        for xi, v in zip(x, s.char_acc):
            ax.text(xi, 100 * v + 3, f"{100 * v:.0f}", ha="center", fontsize=8)
    axes[0].set_ylabel("OCR character accuracy (%)")
    fig.savefig(FIG / "05_real_ocr.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (dataset, df) in zip(axes, frames.items()):
        d = df.acc_text_aware - df.acc_blurred
        acc = df.text_aware_accept.values
        bins = np.linspace(-1, 1, 15)
        ax.hist([d[acc], d[~acc]], bins=bins, stacked=True, color=["#2a6fb0", "#d9d9d9"],
                label=["accepted by gate", "refused by gate"])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(dataset); ax.set_xlabel("OCR accuracy change if restored (text-aware)")
    axes[0].set_ylabel("images"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("What the gate lets through: accuracy change of the restoration it would produce", y=1.02)
    fig.savefig(FIG / "05_gate_real.png"); plt.close(fig)
    print(f"\nwrote {CSV / '05_summary.csv'} and figures 05_*.png")


if __name__ == "__main__":
    main()
