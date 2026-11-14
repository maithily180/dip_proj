"""
Robustness sweep: for each test image, sweep noise level and repeat with
multiple random seeds, to see how quickly angle/length estimation degrades,
and whether low-texture (text/plate-like) images are specifically worse than
textured natural images -- this is the crux question for the proposal.
"""
import numpy as np
from skimage import data, img_as_float
import json
import os
from phase0 import motion_kernel, apply_blur, estimate_angle, estimate_length, OUT

def angle_err(a, b):
    return min(abs(a - b), 180 - abs(a - b))

def sweep(img, name, angle_true=30, length_true=9, noise_levels=(0.0, 0.005, 0.01, 0.02, 0.03, 0.05), n_seeds=5):
    rows = []
    for noise in noise_levels:
        errs = []
        len_errs = []
        for seed in range(n_seeds):
            k = motion_kernel(length_true, angle_true)
            blurred = apply_blur(img, k, noise_sigma=noise, seed=seed)
            a_est, *_ = estimate_angle(blurred)
            l_est, *_ = estimate_length(blurred, a_est)
            errs.append(angle_err(a_est, angle_true))
            if l_est is not None:
                len_errs.append(abs(l_est - length_true))
        row = {
            "image": name, "noise": noise,
            "angle_err_mean": float(np.mean(errs)), "angle_err_max": float(np.max(errs)),
            "frac_angle_err_gt15": float(np.mean(np.array(errs) > 15)),
            "length_err_mean": float(np.mean(len_errs)) if len_errs else None,
        }
        rows.append(row)
        print(f"{name:10s} noise={noise:.3f}  angle_err mean={row['angle_err_mean']:5.1f} max={row['angle_err_max']:5.1f} "
              f"frac>15deg={row['frac_angle_err_gt15']:.2f}  length_err_mean={row['length_err_mean']}")
    return rows

def main():
    imgs = {
        "text": img_as_float(data.text()),
        "camera": img_as_float(data.camera()),
        "checker": img_as_float(data.checkerboard()),
        "coins": img_as_float(data.coins()),
    }
    all_rows = []
    for name, img in imgs.items():
        all_rows += sweep(img, name)
    with open(os.path.join(OUT, "robustness.json"), "w") as f:
        json.dump(all_rows, f, indent=2)

if __name__ == "__main__":
    main()
