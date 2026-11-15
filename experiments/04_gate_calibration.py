"""Experiment 4 - calibrating and validating the confidence gate (objective 4).

The gate is useful only if it refuses the restorations that would make an image
*worse* while letting through the ones that help.  So the label of every case
is the outcome of the restoration it guards:

* synthetic : PSNR of the RL restoration with the *estimated* PSF vs. the
  blurred input (``good`` = higher);
* real      : OCR character accuracy after vs. before restoration
  (``good`` = higher, ``bad`` = lower, unchanged = neutral).

Calibration uses development data only - the synthetic dev split plus the
LPBlur and CCPD-Blur dev subsets, which are disjoint from every image the final
numbers are reported on.  The thresholds are chosen to maximise the number of
helpful restorations kept, subject to the gate letting through harmful ones in
at most ``MAX_HARM`` of the images of *every* development set.  The chosen
thresholds are frozen in ``textdeblur.gate.DEFAULT_GATE``; this script checks
that they match.

Held-out evaluation: synthetic test split (from experiment 3's results) here,
real test images in experiment 5.

Outputs: results/csv/04_*.csv, results/gate_calibration.json, results/figures/04_*.png
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from common import CSV, FIG, RESULTS, pmap, setup_matplotlib
from real_data import ccpd_items, load_item, lpblur_items
from sweep import run_sweep
from textdeblur.deconv import richardson_lucy
from textdeblur.estimate import BASELINE, TEXT_AWARE, estimate_blur
from textdeblur.gate import DEFAULT_GATE, GateConfig, auc, confusion, consistency, roc_curve
from textdeblur.metrics import char_accuracy

CONFIGS = {"baseline": BASELINE, "text_aware": TEXT_AWARE}
SIGNALS = ["confidence", "angular_z", "peak_ratio", "agreement"]
MAX_HARM = 0.05                    # max share of images harmed by an accepted restoration
N_REAL_DEV = 150


def real_dev_row(item):
    it = load_item(item)
    b = it.blurred
    row = dict(dataset=item.dataset, name=item.name, truth=it.truth)
    before = char_accuracy(it.reader(b), it.truth)
    for name, cfg in CONFIGS.items():
        est = estimate_blur(b, cfg)
        after = char_accuracy(it.reader(np.clip(richardson_lucy(b, est.psf(), num_iter=30), 0, 1)), it.truth)
        rows = dict(variant=name, confidence=est.confidence, angular_z=est.angular_z,
                    peak_ratio=est.peak_ratio, agreement=consistency(b, est),
                    angle=est.angle, length=est.length, before=before, after=after)
        row[name] = rows
    return row


def load_dev():
    syn = run_sweep(CONFIGS, split="dev", restore_for=tuple(CONFIGS), agreement_for=tuple(CONFIGS),
                    desc="gate dev (synthetic)")
    syn = syn.assign(dataset="synthetic", gain=syn.psnr_est - syn.psnr_blur)
    syn["good"] = syn.gain > 0
    syn["bad"] = syn.gain < 0
    syn.to_csv(CSV / "04_gate_dev_synthetic.csv", index=False)

    items = lpblur_items(N_REAL_DEV, "dev") + ccpd_items(N_REAL_DEV, "dev")
    raw = pmap(real_dev_row, items, chunksize=2, desc="gate dev (real)")
    rows = []
    for r in raw:
        if r["dataset"] == "lpblur" and len(r["truth"]) != 6:
            continue
        for name in CONFIGS:
            rows.append(dict(dataset=r["dataset"], image=r["name"], **r[name]))
    real = pd.DataFrame(rows)
    real["gain"] = real.after - real.before
    real["good"] = real.gain > 0
    real["bad"] = real.gain < 0
    real.to_csv(CSV / "04_gate_dev_real.csv", index=False)
    return syn, real


def accept_mask(df: pd.DataFrame, g: GateConfig) -> np.ndarray:
    return ((df.confidence >= g.min_confidence) & (df.angular_z >= g.min_angular_z)
            & (df.peak_ratio >= g.min_peak_ratio) & (df.agreement >= g.min_agreement)).values


def evaluate(df: pd.DataFrame, g: GateConfig) -> dict:
    a = accept_mask(df, g)
    return dict(accept=a.mean(), kept_good=(a & df.good.values).sum() / max(df.good.sum(), 1),
                harmed=(a & df.bad.values).mean(), harm_avoided=(~a & df.bad.values).sum() / max(df.bad.sum(), 1),
                good_withheld=(~a & df.good.values).sum() / max(df.good.sum(), 1),
                mean_gain=np.where(a, df.gain.values, 0.0).mean())


def calibrate(sets: dict[str, pd.DataFrame]) -> tuple[GateConfig, pd.DataFrame]:
    pooled = pd.concat(sets.values())
    confs = np.unique(np.round(np.quantile(pooled.confidence, np.linspace(0, 0.9, 19)), 1))
    angzs = np.unique(np.round(np.quantile(pooled.angular_z, np.linspace(0, 0.9, 19)), 1))
    grid = []
    for c, z, ag in itertools.product(np.r_[0.0, confs], np.r_[0.0, angzs], range(5)):
        g = GateConfig(min_confidence=float(c), min_angular_z=float(z), min_agreement=int(ag))
        ev = {k: evaluate(d, g) for k, d in sets.items()}
        grid.append(dict(min_confidence=c, min_angular_z=z, min_agreement=ag,
                         feasible=all(e["harmed"] <= MAX_HARM for e in ev.values()),
                         score=np.mean([e["kept_good"] for e in ev.values()]),
                         **{f"{k}_{m}": e[m] for k, e in ev.items() for m in ("accept", "kept_good", "harmed")}))
    grid = pd.DataFrame(grid)
    feas = grid[grid.feasible].sort_values(["score", "min_agreement", "min_angular_z", "min_confidence"],
                                           ascending=[False, True, True, True])
    best = feas.iloc[0]
    return GateConfig(min_confidence=float(best.min_confidence), min_angular_z=float(best.min_angular_z),
                      min_agreement=int(best.min_agreement)), grid


def main():
    syn, real = load_dev()
    pd.set_option("display.width", 220)

    sets_all = {"synthetic": syn, "lpblur": real[real.dataset == "lpblur"], "ccpd": real[real.dataset == "ccpd"]}
    print("\n== How often does restoring with the estimate help (dev)? ==")
    for k, d in sets_all.items():
        for v in CONFIGS:
            s = d[d.variant == v]
            print(f"  {k:9s} {v:10s} n={len(s):5d}  helps {100 * s.good.mean():5.1f}%  hurts {100 * s.bad.mean():5.1f}%")

    print("\n== Ranking power of each gate signal: ROC AUC for 'restoration helps' (dev, text-aware) ==")
    aucs = []
    for k, d in sets_all.items():
        s = d[d.variant == "text_aware"]
        groups = [(k, s)] + ([(f"synthetic/{c}", s[s.cls == c]) for c in ("natural", "text", "plate")] if k == "synthetic" else [])
        for name, ss in groups:
            r = {"set": name}
            for sig in SIGNALS:
                fpr, tpr, _ = roc_curve(ss[sig].values, ss.good.values)
                r[sig] = auc(fpr, tpr)
            aucs.append(r)
    aucs = pd.DataFrame(aucs)
    aucs.to_csv(CSV / "04_signal_auc.csv", index=False)
    print(aucs.round(3).to_string(index=False))

    sets = {k: d[d.variant == "text_aware"] for k, d in sets_all.items()}
    gate, grid = calibrate(sets)
    grid.to_csv(CSV / "04_gate_grid.csv", index=False)
    print(f"\n== Calibrated gate (harm <= {100 * MAX_HARM:.0f}% of images on every dev set) ==\n  {gate}")
    print(f"  frozen DEFAULT_GATE: {DEFAULT_GATE}  ->  {'MATCH' if gate == DEFAULT_GATE else 'DIFFERENT - update textdeblur/gate.py'}")
    (RESULTS / "gate_calibration.json").write_text(json.dumps(dict(
        max_harm=MAX_HARM, calibrated=gate.__dict__, frozen=DEFAULT_GATE.__dict__), indent=2))

    print("\n== Gate behaviour on dev data (frozen gate) ==")
    rows = []
    for k, d in sets_all.items():
        for v in CONFIGS:
            s = d[d.variant == v]
            for gname, g in (("no gate", GateConfig()), ("gate", DEFAULT_GATE)):
                rows.append(dict(set=k, variant=v, gate=gname, **evaluate(s, g), **confusion(accept_mask(s, g), s.good.values)))
    dev_tab = pd.DataFrame(rows)
    dev_tab.to_csv(CSV / "04_gate_dev_eval.csv", index=False)
    print(dev_tab.round(3).to_string(index=False))

    # --- held-out synthetic evaluation (experiment 3 results)
    test_tab = None
    f3 = CSV / "03_preprocess.csv"
    if f3.exists():
        t = pd.read_csv(f3)
        t = t[t.variant.isin(["gray", "text_aware"]) & t.psnr_est.notna()].copy()
        t["gain"] = t.psnr_est - t.psnr_blur; t["good"] = t.gain > 0; t["bad"] = t.gain < 0
        if "agreement" in t and t.agreement.notna().all():
            rows = []
            for (cls, v), s in t.groupby(["cls", "variant"]):
                for gname, g in (("no gate", GateConfig()), ("gate", DEFAULT_GATE)):
                    rows.append(dict(cls=cls, variant=v, gate=gname, n=len(s), **evaluate(s, g),
                                     **confusion(accept_mask(s, g), s.good.values)))
            test_tab = pd.DataFrame(rows)
            test_tab.to_csv(CSV / "04_gate_test_synthetic.csv", index=False)
            print("\n== Held-out synthetic test split (frozen gate) ==")
            print(test_tab.round(3).to_string(index=False))

    # --- figures
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
    colors = {"confidence": "#8c8c8c", "angular_z": "#2a6fb0", "peak_ratio": "#dd8452", "agreement": "#55a868"}
    names = {"confidence": "dip depth (confidence)", "angular_z": "directional contrast",
             "peak_ratio": "peak ratio", "agreement": "split-half agreement"}
    for ax, (k, s) in zip(axes, sets.items()):
        for sig in SIGNALS:
            fpr, tpr, _ = roc_curve(s[sig].values, s.good.values)
            ax.plot(fpr, tpr, color=colors[sig], lw=1.8, label=f"{names[sig]} ({auc(fpr, tpr):.2f})",
                    drawstyle="steps-post" if sig == "agreement" else "default")
        ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls=":")
        a = accept_mask(s, DEFAULT_GATE)
        ax.plot((a & ~s.good.values).sum() / max((~s.good).sum(), 1), (a & s.good.values).sum() / max(s.good.sum(), 1),
                "k*", ms=12, label="frozen gate")
        ax.set_title(f"{k} (dev, n={len(s)})"); ax.set_xlabel("share of non-helpful restorations accepted")
        ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    axes[0].set_ylabel("share of helpful restorations accepted")
    fig.suptitle("Gate signals: ROC for 'the restoration will help' (AUC in brackets)", y=1.02)
    fig.savefig(FIG / "04_gate_roc.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
    for ax, (k, s) in zip(axes, sets.items()):
        zs = np.linspace(0, np.quantile(s.angular_z, 0.98), 40)
        ev = [evaluate(s, GateConfig(min_confidence=DEFAULT_GATE.min_confidence, min_angular_z=z,
                                     min_agreement=DEFAULT_GATE.min_agreement)) for z in zs]
        ax.plot(zs, [100 * e["harm_avoided"] for e in ev], color="#2a6fb0", lw=2, label="harm avoided")
        ax.plot(zs, [100 * e["good_withheld"] for e in ev], color="#c0392b", lw=2, label="good restorations withheld")
        ax.axvline(DEFAULT_GATE.min_angular_z, color="k", lw=0.8, ls="--")
        ax.set_title(k); ax.set_xlabel("directional-contrast threshold (sigma)")
    axes[0].set_ylabel("% of harmful / helpful cases"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Trade-off of the gate threshold (other tests fixed at their frozen values; dashed: frozen)", y=1.03)
    fig.savefig(FIG / "04_gate_tradeoff.png"); plt.close(fig)
    print("\nwrote results/csv/04_*.csv, results/gate_calibration.json, results/figures/04_*.png")


if __name__ == "__main__":
    main()
