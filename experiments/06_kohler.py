"""Experiment 6 - the Köhler camera-shake benchmark as a real-blur reference.

The benchmark holds 48 photographs (4 scenes x 12 camera trajectories)
recorded by replaying real hand-shake on a hexapod, plus ~199 sharp frames
along each trajectory.  Its kernels are *curved trajectories*, not straight
lines, and the blur is mildly non-uniform, so a two-parameter linear PSF can
only approximate them.  The benchmark is used here exactly as the proposal
states: as a global-kernel reference point for real camera shake.

For every image we report

* the reference kernel, obtained by regularised Fourier division of the
  blurred image by a sharp frame (the kernel that best maps sharp -> blurred);
* how linear that kernel is, and its equivalent angle/length (second moments);
* the estimated line kernel's similarity to the reference kernel;
* the benchmark PSNR (best over the sharp frames after intensity scaling and
  registration, as in the official evaluation code) of
    - the blurred input,
    - Richardson-Lucy with the reference kernel       (ceiling),
    - Richardson-Lucy with the best *line* fit to it  (model ceiling),
    - the baseline and the text-aware pipeline.

Outputs: results/csv/06_kohler.csv, results/figures/06_*.png
"""
from __future__ import annotations

import re
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
from skimage.registration import phase_cross_correlation

from common import CSV, FIG, pmap, setup_matplotlib
from textdeblur.data import DATA, imread, to_gray
from textdeblur.deconv import richardson_lucy
from textdeblur.estimate import BASELINE, TEXT_AWARE, estimate_blur
from textdeblur.gate import DEFAULT_GATE, consistency, decide
from textdeblur.metrics import angle_error, kernel_similarity
from textdeblur.psf import motion_psf

BASE = DATA / "kohler"
KSIZE = 151                 # support of the reference kernel
RL_ITERS = 60


def gt_frames(image: int, kernel: int) -> list:
    pat = re.compile(rf"GroundTruth{image}_{kernel}_(\d+)\.png$")
    return sorted(p for p in (BASE / "gt").glob(f"GroundTruth{image}_{kernel}_*.png") if pat.search(p.name))


def reference_kernel(sharp: np.ndarray, blurred: np.ndarray, size: int = KSIZE) -> np.ndarray:
    """Kernel that maps ``sharp`` to ``blurred`` (regularised Fourier division)."""
    win = np.outer(np.hanning(sharp.shape[0]), np.hanning(sharp.shape[1]))
    X = np.fft.fft2((sharp - sharp.mean()) * win)
    B = np.fft.fft2((blurred - blurred.mean()) * win)
    k = np.real(np.fft.ifft2(np.conj(X) * B / (np.abs(X) ** 2 + 1e-3 * np.mean(np.abs(X) ** 2))))
    k = np.fft.fftshift(k)
    cy, cx = np.unravel_index(int(np.argmax(k)), k.shape)
    h = size // 2
    k = np.roll(k, (k.shape[0] // 2 - cy, k.shape[1] // 2 - cx), axis=(0, 1))
    c0, c1 = k.shape[0] // 2, k.shape[1] // 2
    k = k[c0 - h : c0 + h + 1, c1 - h : c1 + h + 1].copy()
    k[k < 0.08 * k.max()] = 0.0                      # remove the noise floor
    # re-centre on the centroid and normalise
    yy, xx = np.mgrid[:size, :size]
    m = k.sum()
    k = np.roll(k, (int(round(h - (yy * k).sum() / m)), int(round(h - (xx * k).sum() / m))), axis=(0, 1))
    return k / k.sum()


def kernel_moments(k: np.ndarray) -> tuple[float, float, float]:
    """Equivalent (angle, length, linearity) of a kernel from second moments.

    ``length`` is that of the uniform line with the same variance along the
    principal axis; ``linearity`` is 1 for a perfect line and 0 for a disc.
    """
    yy, xx = np.mgrid[: k.shape[0], : k.shape[1]]
    m = k.sum()
    mx, my = (xx * k).sum() / m, (yy * k).sum() / m
    x, y = xx - mx, -(yy - my)                       # y up
    cov = np.array([[(x * x * k).sum(), (x * y * k).sum()], [(x * y * k).sum(), (y * y * k).sum()]]) / m
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, 1]
    angle = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    return angle, float(np.sqrt(12 * vals[1])), float(1 - np.sqrt(max(vals[0], 0) / vals[1]))


def benchmark_psnr(restored: np.ndarray, frames: list[np.ndarray]) -> float:
    """Köhler protocol: best PSNR over the sharp frames, after least-squares
    intensity scaling and integer registration of the restored image."""
    best = -np.inf
    z = restored
    for x in frames:
        zs = z * (np.sum(x * z) / np.sum(z * z))
        shift, _, _ = phase_cross_correlation(x.mean(axis=-1), zs.mean(axis=-1), upsample_factor=1)
        dy, dx = int(shift[0]), int(shift[1])
        zr = np.roll(zs, (dy, dx), axis=(0, 1))
        mask = np.ones(x.shape[:2], bool)
        if dy > 0: mask[:dy] = False
        elif dy < 0: mask[dy:] = False
        if dx > 0: mask[:, :dx] = False
        elif dx < 0: mask[:, dx:] = False
        mse = np.mean(((x - zr) * mask[..., None]) ** 2)
        best = max(best, 10 * np.log10(1.0 / mse))
    return float(best)


@lru_cache(maxsize=None)
def kernel_for(kernel: int):
    """Reference kernel of trajectory ``kernel`` (from the first scene that
    has ground truth; the same trajectory is replayed for every scene)."""
    for image in (1, 2, 3, 4):
        fr = gt_frames(image, kernel)
        if fr:
            sharp = to_gray(imread(fr[len(fr) // 2]))
            blurred = to_gray(imread(BASE / "blurry" / f"Blurry{image}_{kernel}.png"))
            return reference_kernel(sharp, blurred)
    return None


def rl(img, psf):
    return np.clip(richardson_lucy(img, psf, num_iter=RL_ITERS, stop=1.0), 0, 1)


def run(job):
    image, kernel = job
    blurred = imread(BASE / "blurry" / f"Blurry{image}_{kernel}.png")
    frames = [imread(p) for p in gt_frames(image, kernel)]
    kref = kernel_for(kernel)
    row = dict(image=image, kernel=kernel, has_gt=bool(frames))
    if kref is not None:
        a, L, lin = kernel_moments(kref)
        row.update(ref_angle=a, ref_length=L, ref_linearity=lin)

    restored = {}
    for name, cfg in (("baseline", BASELINE), ("text_aware", TEXT_AWARE)):
        est = estimate_blur(blurred, cfg)
        agree = consistency(blurred, est, cfg)
        row.update({f"{name}_angle": est.angle, f"{name}_length": est.length,
                    f"{name}_conf": est.confidence, f"{name}_ratio": est.peak_ratio, f"{name}_agree": agree,
                    f"{name}_accept": decide(est, DEFAULT_GATE, agree).accept})
        if kref is not None:
            row[f"{name}_ksim"] = kernel_similarity(est.psf(), kref)
            row[f"{name}_angle_err"] = angle_error(est.angle, row["ref_angle"])
        restored[name] = rl(blurred, est.psf())
    if kref is not None:
        line = motion_psf(row["ref_length"], row["ref_angle"])
        row["line_ksim"] = kernel_similarity(line, kref)
        restored["ref_kernel"] = rl(blurred, kref)
        restored["line_fit"] = rl(blurred, line)
    if frames:
        row["psnr_blurred"] = benchmark_psnr(blurred, frames)
        for name, img in restored.items():
            row[f"psnr_{name}"] = benchmark_psnr(img, frames)
    return row


def main():
    if "--plot-only" in sys.argv:                  # redraw from a previous run
        df = pd.read_csv(CSV / "06_kohler.csv")
    else:
        jobs = [(i, k) for i in range(1, 5) for k in range(1, 13)]
        df = pd.DataFrame(pmap(run, jobs, workers=8, chunksize=1, desc="Köhler"))
        df.to_csv(CSV / "06_kohler.csv", index=False)
    pd.set_option("display.width", 220)

    print("\n== Reference kernels (one per trajectory) ==")
    k = df[df.image == df.image.min()].set_index("kernel")
    print(k[["ref_angle", "ref_length", "ref_linearity", "line_ksim"]].round(2).to_string())

    print("\n== Kernel similarity of the estimated line kernel to the reference kernel ==")
    print(df[["line_ksim", "baseline_ksim", "text_aware_ksim"]].describe().loc[["mean", "50%", "min", "max"]].round(3).to_string())
    lin = df[df.ref_linearity >= 0.75]
    print(f"\nNear-linear trajectories (linearity >= 0.75): {lin.kernel.nunique()} of 12, {len(lin)} images")
    for name in ("baseline", "text_aware"):
        print(f"  {name:10s} median angle error {lin[f'{name}_angle_err'].median():.1f} deg, "
              f"within 10 deg: {100 * (lin[f'{name}_angle_err'] <= 10).mean():.0f}%, "
              f"kernel similarity {lin[f'{name}_ksim'].mean():.3f} (line-fit ceiling {lin.line_ksim.mean():.3f})")

    g = df[df.has_gt]
    if len(g):
        cols = ["psnr_blurred", "psnr_baseline", "psnr_text_aware", "psnr_line_fit", "psnr_ref_kernel"]
        print(f"\n== Benchmark PSNR (dB), {len(g)} images with ground truth ==")
        print(g[cols].mean().round(2).to_string())
        gated = np.where(g.text_aware_accept, g.psnr_text_aware, g.psnr_blurred)
        print(f"text-aware + gate: {gated.mean():.2f} dB  (accepted {100 * g.text_aware_accept.mean():.0f}% of images)")
        for name in ("baseline", "text_aware"):
            print(f"  {name:10s} worse than blurred input in {100 * (g[f'psnr_{name}'] < g.psnr_blurred).mean():.0f}% of images")
        gl = g[g.ref_linearity >= 0.75]
        if len(gl):
            print(f"near-linear subset ({len(gl)} images):"); print(gl[cols].mean().round(2).to_string())

    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 12, figsize=(16, 3.2))
    for kk in range(1, 13):
        kref = kernel_for(kk)
        r = df[(df.kernel == kk)].iloc[0]
        for row_i, (img, title) in enumerate([
            (kref, f"#{kk}  lin {r.get('ref_linearity', np.nan):.2f}"),
            (motion_psf(min(r.text_aware_length, KSIZE - 4), r.text_aware_angle, size=KSIZE),
             f"L={r.text_aware_length:.0f}, sim {r.get('text_aware_ksim', np.nan):.2f}"),
        ]):
            ax = axes[row_i, kk - 1]
            if img is not None:
                c = KSIZE // 2
                ax.imshow(img[c - 55 : c + 56, c - 55 : c + 56], cmap="magma")
            ax.set_title(title, fontsize=7.5); ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("recorded", fontsize=8); axes[1, 0].set_ylabel("estimated line", fontsize=8)
    fig.suptitle("Köhler benchmark: recorded camera-shake kernels vs. the estimated linear PSF (scene 1)", y=1.04, fontsize=10)
    fig.savefig(FIG / "06_kohler_kernels.png"); plt.close(fig)

    if len(g):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.scatter(g.ref_linearity, g.psnr_text_aware - g.psnr_blurred, c="#2a6fb0", s=26, label="text-aware estimate")
        ax.scatter(g.ref_linearity, g.psnr_ref_kernel - g.psnr_blurred, c="#999999", s=14, marker="^", label="reference kernel (ceiling)")
        ax.scatter(g.ref_linearity, g.psnr_line_fit - g.psnr_blurred, c="#dd8452", s=14, marker="s", label="best line fit to it")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("linearity of the recorded kernel (1 = straight line)")
        ax.set_ylabel("PSNR change vs. blurred input (dB)")
        ax.set_title("Real camera shake: the recorded kernel helps, a straight line does not")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(FIG / "06_kohler_gain_vs_linearity.png"); plt.close(fig)
    print(f"\nwrote results/csv/06_kohler.csv")


if __name__ == "__main__":
    main()
