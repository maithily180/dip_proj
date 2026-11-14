"""
Full comparison sweep: naive Radon-variance angle estimator vs. the improved
joint cepstral-dip-strength estimator, across texture types / angles / noise
levels. Saves everything to JSON so the report can cite exact numbers.
"""
import numpy as np
from skimage import data, img_as_float
import json
import os
from phase0 import motion_kernel, apply_blur, estimate_angle, estimate_angle_length_joint, OUT

def angle_err(a, b):
    return min(abs(a - b), 180 - abs(a - b))

def main():
    imgs = {
        "text (plate-proxy, low-texture)": img_as_float(data.text()),
        "camera (natural photo)": img_as_float(data.camera()),
        "coins (natural photo)": img_as_float(data.coins()),
        "checkerboard (periodic, pathological)": img_as_float(data.checkerboard()),
    }
    angles = (15, 30, 45, 60, 90, 100, 120)
    noises = (0.0, 0.01, 0.02, 0.03, 0.05)
    n_seeds = 5
    length_true = 9

    rows = []
    for name, img in imgs.items():
        for angle_true in angles:
            for noise in noises:
                naive_errs = []
                joint_errs, joint_lerrs = [], []
                for seed in range(n_seeds):
                    k = motion_kernel(length_true, angle_true)
                    b = apply_blur(img, k, noise_sigma=noise, seed=seed)
                    a_naive, *_ = estimate_angle(b)
                    naive_errs.append(angle_err(a_naive, angle_true))
                    a_joint, l_joint, *_ = estimate_angle_length_joint(b)
                    joint_errs.append(angle_err(a_joint, angle_true))
                    joint_lerrs.append(abs(l_joint - length_true))
                rows.append({
                    "image": name, "angle_true": angle_true, "noise": noise,
                    "naive_angle_err_mean": float(np.mean(naive_errs)),
                    "naive_angle_err_max": float(np.max(naive_errs)),
                    "joint_angle_err_mean": float(np.mean(joint_errs)),
                    "joint_angle_err_max": float(np.max(joint_errs)),
                    "joint_length_err_mean": float(np.mean(joint_lerrs)),
                    "joint_length_err_max": float(np.max(joint_lerrs)),
                })
        print(f"done: {name}")

    with open(os.path.join(OUT, "full_sweep.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("saved full_sweep.json")

if __name__ == "__main__":
    main()
