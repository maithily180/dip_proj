"""Experiment 0 - the development-split evidence behind the design choices.

Every choice that shapes the final ("text-aware") system was made here, on
development data only:

* synthetic dev split  - success of each estimator variant on the full angle x
  length x noise grid (plate, text and natural crops);
* LPBlur dev subset    - OCR accuracy after RL restoration with each variant's
  estimate (real motion blur);
* CCPD-Blur dev subset - the same, on mostly non-linear real blur.

The variants are the contrast/edge pre-processing operators the proposal set out
to test and the spectral band limits the analysis led to.  The held-out
numbers in the report come from experiments 2-6, which never touch these images.

Outputs: results/csv/00_design_dev_synthetic.csv, 00_design_dev_real.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import CSV, pmap
from real_data import ccpd_items, load_item, lpblur_items
from sweep import run_sweep
from textdeblur.deconv import richardson_lucy
from textdeblur.estimate import BASELINE, TEXT_AWARE, estimate_blur
from textdeblur.metrics import char_accuracy

B = BASELINE
VARIANTS = {
    "gray": B,
    "clahe": B.with_(preprocess="clahe"),
    "lcn": B.with_(preprocess="lcn"),
    "homomorphic": B.with_(preprocess="homomorphic"),
    "text": B.with_(preprocess="text"),
    "band_auto": B.with_(cutoff="auto"),
    "band0.7": B.with_(cutoff=0.7),
    "band0.5": B.with_(cutoff=0.5),
    "band0.3": B.with_(cutoff=0.3),
    "multiband": TEXT_AWARE,
}
N_REAL = 150


def real_row(item):
    it = load_item(item)
    b = it.blurred
    before = char_accuracy(it.reader(b), it.truth)
    rows = []
    for name, cfg in VARIANTS.items():
        est = estimate_blur(b, cfg)
        after = char_accuracy(it.reader(np.clip(richardson_lucy(b, est.psf(), num_iter=30), 0, 1)), it.truth)
        rows.append(dict(dataset=item.dataset, image=item.name, truth=it.truth, variant=name,
                         angle=est.angle, length=est.length, cutoff=est.cutoff, confidence=est.confidence,
                         angular_z=est.angular_z, before=before, after=after))
    return rows


def main():
    pd.set_option("display.width", 200)
    syn = run_sweep(VARIANTS, split="dev", desc="design dev (synthetic)")
    syn.to_csv(CSV / "00_design_dev_synthetic.csv", index=False)
    tab = (100 * syn.pivot_table(index="variant", columns="cls", values="success")).loc[list(VARIANTS)]
    tab["mean"] = tab.mean(axis=1)
    print("== Synthetic dev split: estimation success (%) =="); print(tab.round(1).to_string())
    print("\n== Synthetic dev split: success (%) by blur length ==")
    print((100 * syn.pivot_table(index="variant", columns="length", values="success")).loc[list(VARIANTS)].round(0).to_string())

    items = lpblur_items(N_REAL, "dev") + ccpd_items(N_REAL, "dev")
    real = pd.DataFrame([r for rows in pmap(real_row, items, chunksize=2, desc="design dev (real)") for r in rows])
    real = real[(real.dataset != "lpblur") | (real.truth.str.len() == 6)]
    real.to_csv(CSV / "00_design_dev_real.csv", index=False)
    for ds, d in real.groupby("dataset"):
        g = d.groupby("variant", sort=False)
        t = pd.DataFrame({"n": g.size(), "before": g.before.mean(), "after": g.after.mean(),
                          "improved_pct": 100 * g.apply(lambda x: (x.after > x.before).mean()),
                          "worse_pct": 100 * g.apply(lambda x: (x.after < x.before).mean()),
                          "median_length": g.length.median()}).loc[list(VARIANTS)]
        print(f"\n== {ds} dev subset: OCR character accuracy, RL-30 with each variant's estimate ==")
        print(t.round(3).to_string())
    print(f"\nwrote {CSV / '00_design_dev_synthetic.csv'} and {CSV / '00_design_dev_real.csv'}")


if __name__ == "__main__":
    main()
