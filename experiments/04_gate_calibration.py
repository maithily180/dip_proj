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
numbers are reported on.  Each candidate gate is scored on every development
set by its *net share*

    sum(gain of accepted cases) / sum(|gain| of all cases)      in [-1, 1],

the net change it lets through as a fraction of everything at stake in that
set (dB or OCR accuracy).  It rewards keeping helpful restorations, charges
every harmful one at its full cost, and is comparable across sets of different
units and difficulty (refusing everything scores 0).  The gate with the best
mean net share over the three sets is frozen in
``textdeblur.gate.DEFAULT_GATE`` (this script checks that they match).  For
comparison the strictest operating point is also reported: the gate that keeps
the most helpful restorations while harming at most ``MAX_HARM`` of the images
of every set.

Held-out evaluation: synthetic test split (from experiment 3's results) here,
real test images in experiment 5.

Outputs: results/csv/04_*.csv, results/gate_calibration.json, results/figures/04_*.png
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from common import CSV, FIG, RESULTS, pmap, setup_matplotlib
from real_data import ccpd_items, load_item, lpblur_items
from sweep import run_sweep
from textdeblur.deconv import richardson_lucy
from textdeblur.estimate import BASELINE, TEXT_AWARE, estimate_blur
from textdeblur.gate import DEFAULT_GATE, STRICT_GATE, GateConfig, auc, confusion, consistency, roc_curve
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


def load_dev(reuse: bool = False):
    f_syn, f_real = CSV / "04_gate_dev_synthetic.csv", CSV / "04_gate_dev_real.csv"
    if reuse and f_syn.exists() and f_real.exists():
        return pd.read_csv(f_syn), pd.read_csv(f_real)
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
                mean_gain=np.where(a, df.gain.values, 0.0).mean(),
                net_share=np.where(a, df.gain.values, 0.0).sum() / max(df.gain.abs().sum(), 1e-12))


def calibrate(sets: dict[str, pd.DataFrame]) -> tuple[GateConfig, GateConfig, pd.DataFrame]:
    """Grid search over the three thresholds.  Returns (best mean net share,
    strict operating point, full grid).  Ties go to the looser gate."""
    q = np.linspace(0, 0.9, 19)
    confs = np.unique(np.round(np.concatenate([np.quantile(d.confidence, q) for d in sets.values()]), 1))
    angzs = np.unique(np.round(np.concatenate([np.quantile(d.angular_z, q) for d in sets.values()]), 1))
    grid = []
    for c, z, ag in itertools.product(np.r_[0.0, confs], np.r_[0.0, angzs], range(5)):
        g = GateConfig(min_confidence=float(c), min_angular_z=float(z), min_agreement=int(ag))
        ev = {k: evaluate(d, g) for k, d in sets.items()}
        grid.append(dict(min_confidence=c, min_angular_z=z, min_agreement=ag,
                         share=np.mean([e["net_share"] for e in ev.values()]),
                         feasible=all(e["harmed"] <= MAX_HARM for e in ev.values()),
                         kept=np.mean([e["kept_good"] for e in ev.values()]),
                         **{f"{k}_{m}": e[m] for k, e in ev.items()
                            for m in ("accept", "kept_good", "harmed", "net_share")}))
    grid = pd.DataFrame(grid)
    order = ["min_agreement", "min_angular_z", "min_confidence"]
    best = grid.sort_values(["share"] + order, ascending=[False, True, True, True]).iloc[0]
    strict = grid[grid.feasible].sort_values(["kept"] + order, ascending=[False, True, True, True]).iloc[0]
    mk = lambda r: GateConfig(min_confidence=float(r.min_confidence), min_angular_z=float(r.min_angular_z),
                              min_agreement=int(r.min_agreement))
    return mk(best), mk(strict), grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="reuse the dev CSVs of a previous run")
    syn, real = load_dev(ap.parse_args().reuse)
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
    gate, strict, grid = calibrate(sets)
    grid.to_csv(CSV / "04_gate_grid.csv", index=False)
    print(f"\n== Calibrated gate (best mean net share over the dev sets) ==\n  {gate}")
    print(f"  frozen DEFAULT_GATE: {DEFAULT_GATE}  ->  {'MATCH' if gate == DEFAULT_GATE else 'DIFFERENT - update textdeblur/gate.py'}")
    print(f"  strict alternative (harm <= {100 * MAX_HARM:.0f}% of images on every dev set): {strict}"
          f"  ->  {'MATCH' if strict == STRICT_GATE else 'DIFFERENT - update STRICT_GATE'}")
    print("\n== Dev operating points (text-aware) ==")
    for name, g in (("no gate", GateConfig()), ("frozen gate", DEFAULT_GATE), ("strict", strict)):
        print(f"  {name:12s}", {k: {m: round(float(v), 3) for m, v in evaluate(d, g).items()
                                    if m in ("accept", "kept_good", "harmed", "net_share")}
                                for k, d in sets.items()})
    (RESULTS / "gate_calibration.json").write_text(json.dumps(dict(
        objective="mean net share over dev sets", calibrated=gate.__dict__, frozen=DEFAULT_GATE.__dict__,
        strict_max_harm=MAX_HARM, strict=strict.__dict__), indent=2))

    print("\n== Gate behaviour on dev data ==")
    rows = []
    for k, d in sets_all.items():
        for v in CONFIGS:
            s = d[d.variant == v]
            for gname, g in (("no gate", GateConfig()), ("gate", DEFAULT_GATE), ("strict", strict)):
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
                for gname, g in (("no gate", GateConfig()), ("gate", DEFAULT_GATE), ("strict", strict)):
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

    # trade-off: harm avoided vs. good withheld while sweeping the dip-depth
    # threshold, one curve per agreement requirement
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
    ag_colors = {0: "#bbbbbb", 1: "#8fb3d9", 2: "#2a6fb0", 3: "#1b3f66"}
    for ax, (k, s) in zip(axes, sets.items()):
        cs = np.r_[0.0, np.quantile(s.confidence, np.linspace(0, 0.98, 40))]
        for ag, col in ag_colors.items():
            ev = [evaluate(s, GateConfig(min_confidence=float(c), min_agreement=ag)) for c in cs]
            ax.plot([100 * e["good_withheld"] for e in ev], [100 * e["harm_avoided"] for e in ev],
                    color=col, lw=1.8, label=f">= {ag} of 4 halves agree")
        for g, mk, lab in ((DEFAULT_GATE, "*", "frozen gate"), (STRICT_GATE, "s", "strict gate")):
            e = evaluate(s, g)
            ax.plot(100 * e["good_withheld"], 100 * e["harm_avoided"], mk, color="#c0392b", ms=11 if mk == "*" else 7,
                    label=lab)
        ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
        ax.set_title(f"{k} (dev)"); ax.set_xlabel("helpful restorations withheld (%)")
    axes[0].set_ylabel("harmful restorations avoided (%)"); axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("Gate trade-off: sweeping the dip-depth threshold for each agreement requirement", y=1.02)
    fig.savefig(FIG / "04_gate_tradeoff.png"); plt.close(fig)
    print("\nwrote results/csv/04_*.csv, results/gate_calibration.json, results/figures/04_*.png")


if __name__ == "__main__":
    main()
