"""
Phase 0 sanity check for the "Radon + cepstrum + Richardson-Lucy" blind
motion-deblurring pipeline.

Goal: with a KNOWN synthetic blur kernel (angle, length), check whether we can
recover the angle and length well enough from the blurred image alone, and
whether feeding that estimated PSF into Richardson-Lucy actually sharpens the
image. This is the make-or-break step before committing to the project.

Everything here uses only numpy / scipy / scikit-image (no exotic deps).
"""
import numpy as np
from scipy import ndimage
from skimage import data, img_as_float
from skimage.transform import radon
from skimage.restoration import richardson_lucy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Synthetic motion blur kernel + convolution
# ---------------------------------------------------------------------------
def motion_kernel(length, angle_deg, size=None):
    """Line kernel of given length (px) and angle (deg, 0=horizontal, CCW)."""
    length = max(1, int(round(length)))
    if size is None:
        size = length + 4 - (length + 4) % 2 + 1  # odd size, some margin
    size = max(size, length + 5)
    if size % 2 == 0:
        size += 1
    k = np.zeros((size, size), dtype=np.float64)
    c = size // 2
    theta = np.deg2rad(angle_deg)
    # draw a sub-pixel line through the center using many samples
    n_samples = length * 10
    for t in np.linspace(-length / 2, length / 2, n_samples):
        x = c + t * np.cos(theta)
        y = c - t * np.sin(theta)  # image y grows downward
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        for dx in (0, 1):
            for dy in (0, 1):
                xi, yi = x0 + dx, y0 + dy
                if 0 <= xi < size and 0 <= yi < size:
                    wx = 1 - abs(x - xi)
                    wy = 1 - abs(y - yi)
                    if wx > 0 and wy > 0:
                        k[yi, xi] += wx * wy
    k /= k.sum()
    return k


def apply_blur(img, kernel, noise_sigma=0.0, seed=0):
    blurred = ndimage.convolve(img, kernel, mode="reflect")
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        blurred = blurred + rng.normal(0, noise_sigma, blurred.shape)
        blurred = np.clip(blurred, 0, 1)
    return blurred


# ---------------------------------------------------------------------------
# 2. Angle estimation via Radon transform of the log-magnitude spectrum
# ---------------------------------------------------------------------------
def hann2d(shape):
    wy = np.hanning(shape[0])
    wx = np.hanning(shape[1])
    return np.outer(wy, wx)


def estimate_angle(img):
    """Return estimated blur angle in degrees (0-180, our convention)."""
    win = hann2d(img.shape)
    windowed = (img - img.mean()) * win
    F = np.fft.fftshift(np.fft.fft2(windowed))
    mag = np.log1p(np.abs(F))
    # mask DC spike so it doesn't dominate the spectrum
    h, w = mag.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mag_masked = mag.copy()
    mag_masked[r < 3] = mag_masked.mean()  # kill DC spike

    thetas = np.linspace(0.0, 180.0, 181, endpoint=False)
    sino = radon(mag_masked, theta=thetas, circle=True)
    # For each angle, score = variance of the projection profile.
    # Dark stripes in the spectrum run PERPENDICULAR to the blur direction;
    # a Radon projection taken PARALLEL to the stripes (i.e. at the stripe
    # angle) integrates along constant-intensity lines -> low variance.
    # Empirically (verified against ground truth in main()) the angle that
    # minimizes variance corresponds to (blur_angle + 90) mod 180.
    variances = sino.var(axis=0)
    best_idx = np.argmin(variances)
    stripe_angle = thetas[best_idx]
    blur_angle = (stripe_angle - 90.0) % 180.0
    return blur_angle, thetas, variances, mag_masked


# ---------------------------------------------------------------------------
# 3. Length estimation via 2D cepstrum
# ---------------------------------------------------------------------------
def cepstrum(img):
    win = hann2d(img.shape)
    windowed = (img - img.mean()) * win
    F = np.fft.fft2(windowed)
    logspec = np.log1p(np.abs(F))
    ceps = np.real(np.fft.ifft2(logspec))
    ceps = np.fft.fftshift(ceps)
    return ceps


def estimate_length(img, angle_deg, min_len=3, max_len=60):
    ceps = cepstrum(img)
    h, w = ceps.shape
    cy, cx = h // 2, w // 2

    # sample the cepstrum along the line through the center at `angle_deg`
    max_r = min(cy, cx) - 2
    rs = np.arange(0, max_r)
    theta = np.deg2rad(angle_deg)
    xs = cx + rs * np.cos(theta)
    ys = cy - rs * np.sin(theta)
    profile = ndimage.map_coordinates(ceps, [ys, xs], order=1)

    search = profile[min_len:max_len]
    if len(search) == 0:
        return None, ceps, profile
    rel_idx = np.argmin(search)  # most negative dip = blur length echo
    length = min_len + rel_idx
    return length, ceps, profile


def estimate_angle_length_joint(img, min_len=3, max_len=40, angle_step=1.0):
    """
    More robust replacement for estimate_angle()+estimate_length(): instead of
    picking the angle from Radon-projection variance (fragile, see Phase-0
    writeup), search jointly over candidate angles and, for each, find the
    strongest (most negative) cepstral dip in [min_len, max_len]. The
    (angle, length) pair with the deepest dip wins. This matches the
    intuition that the TRUE blur direction is where the cepstral echo is
    cleanest/strongest, not just where the raw FFT projection has low
    variance.
    """
    ceps = cepstrum(img)
    h, w = ceps.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx) - 2
    rs = np.arange(0, max_r)

    best_angle, best_length, best_depth, best_profile = None, None, -np.inf, None
    for angle in np.arange(0.0, 180.0, angle_step):
        theta = np.deg2rad(angle)
        xs = cx + rs * np.cos(theta)
        ys = cy - rs * np.sin(theta)
        profile = ndimage.map_coordinates(ceps, [ys, xs], order=1)
        seg = profile[min_len:max_len]
        if len(seg) == 0:
            continue
        idx = int(np.argmin(seg))
        depth = -seg[idx]
        if depth > best_depth:
            best_depth = depth
            best_angle = angle
            best_length = min_len + idx
            best_profile = profile
    return best_angle, best_length, ceps, best_profile


# ---------------------------------------------------------------------------
# 4. Deconvolution
# ---------------------------------------------------------------------------
def deblur(img, angle_deg, length_px, iterations=30):
    k = motion_kernel(length_px, angle_deg)
    eps = 1e-3
    restored = richardson_lucy(img, k, num_iter=iterations, clip=False, filter_epsilon=eps)
    return restored, k


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def psnr(a, b, border=0):
    a = np.clip(a, 0, 1)
    b = np.clip(b, 0, 1)
    if border > 0:
        a = a[border:-border, border:-border]
        b = b[border:-border, border:-border]
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def run_case(name, sharp, angle_true, length_true, noise_sigma, iterations=30):
    k_true = motion_kernel(length_true, angle_true)
    blurred = apply_blur(sharp, k_true, noise_sigma=noise_sigma)

    angle_est_naive, thetas, variances, spec = estimate_angle(blurred)
    angle_est, length_est, ceps, profile = estimate_angle_length_joint(blurred)

    angle_err = min(abs(angle_est - angle_true), 180 - abs(angle_est - angle_true))
    length_err = None
    restored = None
    border = int(length_true) + 8  # exclude FFT-deconv boundary wraparound artifacts
    psnr_blurred = psnr(blurred, sharp, border=border)
    psnr_restored_true_psf = None
    psnr_restored_est_psf = None

    # restoration using the TRUE psf (upper bound on what deconvolution can do)
    restored_true, _ = deblur(blurred, angle_true, length_true, iterations)
    psnr_restored_true_psf = psnr(restored_true, sharp, border=border)

    if length_est is not None:
        length_err = abs(length_est - length_true)
        restored, k_est = deblur(blurred, angle_est, length_est, iterations)
        psnr_restored_est_psf = psnr(restored, sharp, border=border)

    result = {
        "name": name,
        "angle_true": angle_true,
        "length_true": length_true,
        "noise_sigma": noise_sigma,
        "angle_est": float(angle_est),
        "angle_err_deg": float(angle_err),
        "length_est": None if length_est is None else int(length_est),
        "length_err_px": None if length_err is None else float(length_err),
        "psnr_blurred": float(psnr_blurred),
        "psnr_restored_true_psf": float(psnr_restored_true_psf),
        "psnr_restored_est_psf": None if psnr_restored_est_psf is None else float(psnr_restored_est_psf),
    }

    # save intermediate images for visual inspection
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes[0, 0].imshow(sharp, cmap="gray"); axes[0, 0].set_title("sharp (ground truth)")
    axes[0, 1].imshow(blurred, cmap="gray"); axes[0, 1].set_title(f"blurred\nangle={angle_true} len={length_true} noise={noise_sigma}")
    axes[0, 2].imshow(spec, cmap="gray"); axes[0, 2].set_title("log-magnitude spectrum")
    axes[0, 3].plot(thetas, variances); axes[0, 3].axvline((angle_true + 90) % 180, color="g", ls="--", label="true stripe angle")
    axes[0, 3].axvline((angle_est_naive + 90) % 180, color="r", ls=":", label="naive-variance est (fragile, unused)")
    axes[0, 3].set_title("Radon variance (OLD naive method, for comparison)"); axes[0, 3].legend(fontsize=6)

    axes[1, 0].imshow(ceps, cmap="gray", vmin=-0.05, vmax=0.05); axes[1, 0].set_title("2D cepstrum")
    axes[1, 1].plot(profile); axes[1, 1].axvline(length_true, color="g", ls="--", label="true length")
    if length_est is not None:
        axes[1, 1].axvline(length_est, color="r", ls=":", label="est length")
    axes[1, 1].legend(fontsize=7); axes[1, 1].set_title("cepstrum profile along est. angle")

    axes[1, 2].imshow(restored_true, cmap="gray"); axes[1, 2].set_title(f"RL restore w/ TRUE psf\nPSNR={psnr_restored_true_psf:.1f} dB")
    if restored is not None:
        axes[1, 3].imshow(restored, cmap="gray")
        axes[1, 3].set_title(f"RL restore w/ ESTIMATED psf\nangle_err={angle_err:.1f} len_err={length_err}\nPSNR={psnr_restored_est_psf:.1f} dB")
    else:
        axes[1, 3].set_title("length estimation failed")
    for ax in axes.flat:
        ax.set_xticks([]) if ax not in (axes[0, 3], axes[1, 1]) else None
        ax.set_yticks([]) if ax not in (axes[0, 3], axes[1, 1]) else None
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=110)
    plt.close(fig)

    return result


def main():
    sharp_text = img_as_float(data.text())          # low-contrast text image (plate proxy)
    sharp_camera = img_as_float(data.camera())       # generic natural image
    sharp_checker = img_as_float(data.checkerboard())

    cases = []
    configs = [
        # (image, name, angle, length, noise)
        (sharp_text, "text_a0_l9_clean", 0, 9, 0.0),
        (sharp_text, "text_a30_l9_clean", 30, 9, 0.0),
        (sharp_text, "text_a45_l15_clean", 45, 15, 0.0),
        (sharp_text, "text_a90_l12_clean", 90, 12, 0.0),
        (sharp_text, "text_a30_l9_noisy", 30, 9, 0.01),
        (sharp_text, "text_a30_l9_noisier", 30, 9, 0.03),
        (sharp_text, "text_a30_l9_veryNoisy", 30, 9, 0.05),
        (sharp_text, "text_a120_l20_clean", 120, 20, 0.0),
        (sharp_camera, "camera_a30_l9_clean", 30, 9, 0.0),
        (sharp_camera, "camera_a45_l15_noisy", 45, 15, 0.01),
        (sharp_checker, "checker_a30_l9_clean", 30, 9, 0.0),
    ]

    results = []
    for img, name, angle, length, noise in configs:
        print(f"Running {name} ...")
        r = run_case(name, img, angle, length, noise)
        results.append(r)
        print(f"   angle_true={angle} angle_est={r['angle_est']:.1f} err={r['angle_err_deg']:.1f}deg | "
              f"length_true={length} length_est={r['length_est']} err={r['length_err_px']} | "
              f"PSNR blurred={r['psnr_blurred']:.1f} restored(true psf)={r['psnr_restored_true_psf']:.1f} "
              f"restored(est psf)={r['psnr_restored_est_psf']}")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone. Panels + results.json written to", OUT)


if __name__ == "__main__":
    main()
