"""Experiment 7 - qualitative figures (deliverable D3).

* ``07_worked_example.png``  - every stage of the pipeline on one synthetic plate:
  sharp / blurred, log-spectrum, the (direction x lag) Radon-cepstrum map with
  the detected dip, the profile along the detected direction, restoration.
* ``07_gallery_lpblur.png``  - real LPBlur plates: clear successes, typical
  cases, gate refusals and failures (selected by rule from the experiment-5
  results, not by hand).
* ``07_gallery_ccpd.png``    - CCPD-Blur: what the gate refuses and what it lets
  through.

Run after experiment 5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import CSV, FIG, Source, make_trial, setup_matplotlib, sources
from real_data import LPBLUR, RealItem, load_item
from textdeblur.data import DATA
from textdeblur.deconv import richardson_lucy
from textdeblur.estimate import BASELINE, TEXT_AWARE, estimate_blur
from textdeblur.gate import DEFAULT_GATE, consistency, decide
from textdeblur.metrics import psnr

RL_ITERS = 30


def rl(img, psf, n=RL_ITERS, stop=None):
    return np.clip(richardson_lucy(img, psf, num_iter=n, stop=stop), 0, 1)


def worked_example(plt):
    src: Source = next(s for s in sources() if s.cls == "plate" and s.split == "test")
    length, angle, sigma = 15, 30.0, 0.01
    sharp, blurred = make_trial(src, length, angle, sigma)
    est = estimate_blur(blurred, TEXT_AWARE.with_(max_length=40))
    restored = rl(blurred, est.psf(), 100, stop=1.0)

    fig = plt.figure(figsize=(13, 6.3))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.42, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(sharp, cmap="gray", vmin=0, vmax=1)
    ax.set_title("sharp plate (unknown to the method)")
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(blurred, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"blurred: L={length} px, {angle:.0f} deg, noise {sigma}\nPSNR {psnr(blurred, sharp, 8):.1f} dB")
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(restored, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"restored with estimate (RL)\nPSNR {psnr(restored, sharp, 8):.1f} dB")
    ax = fig.add_subplot(gs[0, 3])
    k = est.psf(); ax.imshow(k, cmap="magma")
    ax.set_title(f"estimated PSF\nL={est.length:.1f} px, {est.angle:.1f} deg")
    for a in fig.axes:
        a.set_xticks([]); a.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    n = est.spectrum.shape[0]
    c = n // 2; r = int(n // 2 * est.cutoff)
    ax.imshow(est.spectrum[c - r : c + r, c - r : c + r], cmap="gray")
    ax.set_title(f"log-spectrum (band limit {est.cutoff:.2f} Nyquist)\ndark stripes = sinc nulls")
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1:3])
    lo = int(np.ceil(est.config.min_length))
    v = float(np.percentile(np.abs(est.profiles[:, lo + 2 :]), 99.5))
    im = ax.imshow(est.profiles[:, lo:].T, aspect="auto", cmap="RdBu", origin="lower", vmin=-v, vmax=v,
                   extent=[0, 180, lo, est.profiles.shape[1]])
    ax.plot(est.angle, est.length, "o", mfc="none", mec="k", ms=14, mew=1.6)
    ax.set_xlabel("projection direction (deg)"); ax.set_ylabel("cepstral lag (px)")
    ax.set_title(f"Radon-cepstrum map: deepest dip = (angle, length)\n"
                 f"confidence {est.confidence:.1f} sigma, directional {est.angular_z:.1f} sigma")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_ticks([]); cb.set_label("cepstrum (red < 0)", fontsize=8)

    ax = fig.add_subplot(gs[1, 3])
    p = est.profile
    ax.plot(np.arange(len(p)), p, color="#2a6fb0")
    ax.axvline(est.length, color="#c0392b", lw=1, ls="--")
    ax.axvline(length, color="k", lw=0.8, ls=":")
    ax.set_xlim(lo, len(p) - 1); ax.set_xlabel("lag (px)")
    ax.set_ylim(p[lo:].min() * 1.3, max(p[lo + 2 :].max() * 1.5, 1e-3))
    ax.set_title("profile along the detected direction\n(red: estimate, dotted: truth)")
    fig.savefig(FIG / "07_worked_example.png"); plt.close(fig)


def _pick(df: pd.DataFrame, k: int = 2) -> list[tuple[str, pd.Series]]:
    """Gallery rows chosen by rule (each image at most once): the largest
    accepted gains, typical accepted gains, the worst and the best refusal,
    and the worst accepted restoration."""
    df = df.assign(d=df.acc_text_aware - df.acc_blurred)
    used: set[str] = set()
    out: list[tuple[str, pd.Series]] = []

    def take(tag, rows, n, cond=lambda r: True):
        for _, r in rows.iterrows():
            if n == 0:
                break
            if r["name"] in used or not cond(r):
                continue
            used.add(r["name"]); out.append((tag, r)); n -= 1

    acc, ref = df[df.text_aware_accept], df[~df.text_aware_accept]
    take("success", acc.sort_values(["d", "acc_text_aware"], ascending=False), k, lambda r: r.d > 0)
    take("typical", acc[(acc.d > 0) & (acc.d < 0.5)], k)
    take("refused (would hurt)", ref.sort_values("d"), 1, lambda r: r.d < 0)
    take("refused (would help)", ref.sort_values("d", ascending=False), 1, lambda r: r.d > 0)
    take("failure", acc.sort_values("d"), 1, lambda r: r.d < 0)
    return out


def real_gallery(plt, dataset: str, df: pd.DataFrame, fname: str):
    picks = _pick(df)
    has_sharp = dataset == "lpblur"
    cols = ["blurred input", "classical baseline", "text-aware + gate"] + (["sharp reference"] if has_sharp else [])
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(3.0 * len(cols), 1.45 * len(picks) + 0.4))
    axes = np.atleast_2d(axes)
    for i, (tag, r) in enumerate(picks):
        path = str(LPBLUR / "blur" / r["name"]) if has_sharp else _ccpd_path(r["name"])
        it = load_item(RealItem(dataset, r["name"], path))
        b = it.blurred
        e0 = estimate_blur(b, BASELINE); e1 = estimate_blur(b, TEXT_AWARE)
        agree = consistency(b, e1) if DEFAULT_GATE.needs_agreement else None
        dec = decide(e1, DEFAULT_GATE, agree)
        imgs = [b, rl(b, e0.psf()), rl(b, e1.psf()) if dec.accept else b] + ([it.sharp] if has_sharp else [])
        texts = [r.ocr_blurred, r.ocr_baseline, r.ocr_text_aware_gated] + ([r.truth] if has_sharp else [])
        for j, (im, t) in enumerate(zip(imgs, texts)):
            ax = axes[i, j]
            show = im if im.ndim == 2 else im[..., :3]
            ax.imshow(show, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            t = t if isinstance(t, str) and t else "-"
            ok = "" if j == len(cols) - 1 and has_sharp else f"  ({100 * _acc(t, r.truth):.0f}%)"
            label = f"{t}{ok}"
            if j == 2 and not dec.accept:
                label = "REFUSED -> input kept\n" + label
            ax.set_xlabel(label, fontsize=7.5)
            if i == 0:
                ax.set_title(cols[j], fontsize=9)
        axes[i, 0].set_ylabel(tag, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / fname); plt.close(fig)


def _acc(text, truth):
    from textdeblur.metrics import char_accuracy
    return char_accuracy(text, truth)


def _ccpd_path(name: str) -> str:
    return str(DATA / "ccpd" / "ccpd_blur" / name)


def main():
    plt = setup_matplotlib()
    worked_example(plt)
    for dataset, fname in (("lpblur", "07_gallery_lpblur.png"), ("ccpd", "07_gallery_ccpd.png")):
        f = CSV / f"05_real_{dataset}.csv"
        if f.exists():
            real_gallery(plt, dataset, pd.read_csv(f, keep_default_na=False), fname)
    print("wrote results/figures/07_*.png")


if __name__ == "__main__":
    main()
