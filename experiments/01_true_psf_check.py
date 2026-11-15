"""Experiment 1 - non-blind restoration with the TRUE PSF.

Foundation check (plan milestone 1): if deconvolving with the exact kernel
does not improve the image, nothing downstream can.  The experiment also
settles two engineering choices:

* boundary handling  - our unknown-boundary Richardson-Lucy against a plain
  FFT-periodic Richardson-Lucy (scikit-image), full frame and interior only;
* iteration count    - PSNR/SSIM as a function of RL iterations per noise level;
* RL vs. Wiener      - the secondary restorer at several noise-to-signal ratios.

Outputs: results/csv/01_true_psf.csv, results/figures/01_*.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.restoration import richardson_lucy as skimage_rl

from common import CSV, FIG, LENGTHS, NOISES, make_trial, pmap, setup_matplotlib, sources
from textdeblur.deconv import richardson_lucy, wiener
from textdeblur.metrics import psnr, ssim
from textdeblur.psf import motion_psf

ITERS = (5, 10, 20, 30, 50, 100, 200)
NSRS = (0.001, 0.003, 0.01, 0.03, 0.1)
ANGLES = (0.0, 45.0, 112.0)
BORDER = 12


def trial(job):
    src, length, angle, sigma = job
    sharp, blurred = make_trial(src, length, angle, sigma)
    psf = motion_psf(length, angle)
    base = dict(cls=src.cls, image=src.name, length=length, angle=angle, sigma=sigma)
    rows = [dict(base, method="blurred", param=0, psnr=psnr(blurred, sharp), ssim=ssim(blurred, sharp),
                 psnr_in=psnr(blurred, sharp, BORDER))]
    for it in ITERS:
        r = richardson_lucy(blurred, psf, num_iter=it)
        rows.append(dict(base, method="rl", param=it, psnr=psnr(r, sharp), ssim=ssim(r, sharp),
                         psnr_in=psnr(r, sharp, BORDER)))
    for tau in (1.0, 1.05, 1.1, 1.2):
        r = richardson_lucy(blurred, psf, num_iter=200, stop=tau)
        rows.append(dict(base, method="rl_auto", param=tau, psnr=psnr(r, sharp), ssim=ssim(r, sharp),
                         psnr_in=psnr(r, sharp, BORDER)))
    r = skimage_rl(np.clip(blurred, 1e-4, 1), psf, num_iter=30, clip=False)
    rows.append(dict(base, method="rl_periodic", param=30, psnr=psnr(r, sharp), ssim=ssim(r, sharp),
                     psnr_in=psnr(r, sharp, BORDER)))
    for nsr in NSRS:
        r = wiener(blurred, psf, nsr=nsr)
        rows.append(dict(base, method="wiener", param=nsr, psnr=psnr(r, sharp), ssim=ssim(r, sharp),
                         psnr_in=psnr(r, sharp, BORDER)))
    return rows


def main():
    srcs = [s for s in sources() if s.split == "dev"]
    jobs = [(s, L, a, n) for s in srcs for L in LENGTHS for a in ANGLES for n in NOISES]
    rows = [r for chunk in pmap(trial, jobs, desc="true-PSF restoration") for r in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(CSV / "01_true_psf.csv", index=False)

    blurred = df[df.method == "blurred"].set_index(["cls", "image", "length", "angle", "sigma"])
    df = df.join(blurred[["psnr", "ssim"]].rename(columns={"psnr": "psnr_b", "ssim": "ssim_b"}),
                 on=["cls", "image", "length", "angle", "sigma"])
    df["gain"] = df.psnr - df.psnr_b
    df["ssim_gain"] = df.ssim - df.ssim_b

    print("\nMean PSNR gain over the blurred input (dB), true PSF")
    rl = df[df.method == "rl"].pivot_table(index="sigma", columns="param", values="gain")
    print(rl.round(2).to_string())
    au = df[df.method == "rl_auto"].pivot_table(index="sigma", columns="param", values="gain")
    print("\nRL with discrepancy-principle stopping, by tau"); print(au.round(2).to_string())
    wn = df[df.method == "wiener"].pivot_table(index="sigma", columns="param", values="gain")
    print("\nWiener, by NSR"); print(wn.round(2).to_string())

    rl30 = df[(df.method == "rl") & (df.param == 30)]
    per = df[df.method == "rl_periodic"]
    print("\nBoundary handling at 30 iterations (mean PSNR, dB)")
    print(f"  blurred input            full {blurred.psnr.mean():.2f}   interior {blurred.psnr_in.mean():.2f}")
    print(f"  periodic RL (skimage)    full {per.psnr.mean():.2f}   interior {per.psnr_in.mean():.2f}")
    print(f"  unknown-boundary RL      full {rl30.psnr.mean():.2f}   interior {rl30.psnr_in.mean():.2f}")
    frac = (rl30.gain > 0).mean()
    print(f"\nRL(30) with the true PSF improves PSNR in {100 * frac:.1f}% of {len(rl30)} trials "
          f"(SSIM improves in {100 * (rl30.ssim_gain > 0).mean():.1f}%)")

    plt = setup_matplotlib()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for sigma, g in df[df.method == "rl"].groupby("sigma"):
        m = g.groupby("param").gain.mean()
        ax[0].plot(m.index, m.values, "o-", label=f"noise sigma = {sigma:g}")
    ax[0].set_xscale("log"); ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("Richardson-Lucy iterations"); ax[0].set_ylabel("PSNR gain over blurred input (dB)")
    ax[0].set_title("RL with the true PSF: iterations vs. noise"); ax[0].legend(frameon=False, fontsize=8)

    for sigma, g in df[df.method == "wiener"].groupby("sigma"):
        m = g.groupby("param").gain.mean()
        ax[1].plot(m.index, m.values, "s-", label=f"noise sigma = {sigma:g}")
    ax[1].set_xscale("log"); ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("Wiener noise-to-signal ratio"); ax[1].set_ylabel("PSNR gain (dB)")
    ax[1].set_title("Wiener with the true PSF"); ax[1].legend(frameon=False, fontsize=8)

    labels = ["blurred\ninput", "periodic RL\n(FFT wrap-around)", "unknown-boundary\nRL (ours)"]
    full = [blurred.psnr.mean(), per.psnr.mean(), rl30.psnr.mean()]
    inner = [blurred.psnr_in.mean(), per.psnr_in.mean(), rl30.psnr_in.mean()]
    x = np.arange(3)
    ax[2].bar(x - 0.2, full, 0.4, label="full frame", color="#4c72b0")
    ax[2].bar(x + 0.2, inner, 0.4, label="interior only", color="#dd8452")
    ax[2].set_xticks(x, labels); ax[2].set_ylabel("mean PSNR (dB)")
    ax[2].set_ylim(min(full + inner) - 3, max(full + inner) + 1.5)
    ax[2].set_title("Boundary handling (30 iterations)"); ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "01_true_psf.png"); plt.close(fig)
    print(f"\nwrote results/csv/01_true_psf.csv and results/figures/01_true_psf.png")


if __name__ == "__main__":
    main()
