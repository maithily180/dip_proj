"""
Simple, no-jargon before/after comparison: blurry input vs. deblurred output.
"""
import numpy as np
from skimage import data, img_as_float
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from phase0 import motion_kernel, apply_blur, estimate_angle_length_joint, deblur

OUT = os.path.join(os.path.dirname(__file__), "out")

cases = [
    ("Text / plate-like image", img_as_float(data.text()), 30, 9, 0.0),
    ("Text / plate-like image (with camera noise)", img_as_float(data.text()), 30, 9, 0.01),
    ("Natural photo", img_as_float(data.camera()), 45, 15, 0.01),
]

fig, axes = plt.subplots(len(cases), 3, figsize=(14, 4.2 * len(cases)))

for row, (label, sharp, angle, length, noise) in enumerate(cases):
    k = motion_kernel(length, angle)
    blurred = apply_blur(sharp, k, noise_sigma=noise, seed=0)
    angle_est, length_est, *_ = estimate_angle_length_joint(blurred)
    restored, _ = deblur(blurred, angle_est, length_est, iterations=30)

    border = length + 8
    crop = lambda im: np.clip(im, 0, 1)[border:-border, border:-border]

    axes[row, 0].imshow(crop(sharp), cmap="gray")
    axes[row, 0].set_title("ORIGINAL (sharp)", fontsize=12, fontweight="bold")

    axes[row, 1].imshow(crop(blurred), cmap="gray")
    axes[row, 1].set_title("INPUT (blurry photo)", fontsize=12, fontweight="bold", color="darkred")

    axes[row, 2].imshow(crop(restored), cmap="gray")
    axes[row, 2].set_title("OUTPUT (after deblurring)", fontsize=12, fontweight="bold", color="darkgreen")

    for ax in axes[row]:
        ax.set_xticks([]); ax.set_yticks([])
    axes[row, 0].set_ylabel(label, fontsize=11, fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "before_after.png"), dpi=130)
print("saved before_after.png")
