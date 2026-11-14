import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.path.join(os.path.dirname(__file__), "out")
rows = json.load(open(os.path.join(OUT, "full_sweep.json")))

imgs = ["text (plate-proxy, low-texture)", "camera (natural photo)",
        "coins (natural photo)", "checkerboard (periodic, pathological)"]
angles = sorted(set(r["angle_true"] for r in rows))
noises = sorted(set(r["noise"] for r in rows))

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for ax, name in zip(axes, imgs):
    grid = np.zeros((len(angles), len(noises)))
    for i, a in enumerate(angles):
        for j, n in enumerate(noises):
            r = next(r for r in rows if r["image"] == name and r["angle_true"] == a and r["noise"] == n)
            grid[i, j] = r["joint_angle_err_mean"]
    im = ax.imshow(grid, vmin=0, vmax=60, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(noises))); ax.set_xticklabels([f"{n:.2f}" for n in noises])
    ax.set_yticks(range(len(angles))); ax.set_yticklabels(angles)
    ax.set_xlabel("noise sigma"); ax.set_ylabel("true blur angle (deg)")
    ax.set_title(name, fontsize=9)
    for i in range(len(angles)):
        for j in range(len(noises)):
            ax.text(j, i, f"{grid[i,j]:.0f}", ha="center", va="center", fontsize=7,
                     color="black" if grid[i, j] < 30 else "white")
fig.colorbar(im, ax=axes, label="mean angle error (deg)", shrink=0.8)
fig.suptitle("Angle estimation error (joint cepstral-dip method) vs. blur angle and noise level", y=1.05)
fig.savefig(os.path.join(OUT, "summary_heatmap.png"), dpi=120, bbox_inches="tight")
print("saved summary_heatmap.png")
