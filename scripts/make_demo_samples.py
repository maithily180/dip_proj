"""Write the small sample images used by the Streamlit demo (``demo/samples``).

Only CCPD (MIT licence) and rendered text are used, so the samples can be
redistributed with the code.  Requires the CCPD subset (see README).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from textdeblur.blur import add_noise, convolve  # noqa: E402
from textdeblur.data import imread, imwrite, list_ccpd, render_text_page, stable_subset  # noqa: E402
from textdeblur.psf import motion_psf  # noqa: E402

OUT = ROOT / "demo" / "samples"


def plate_crop(rec, mx=0.35, my=0.9):
    img = imread(rec.path)
    x1, y1, x2, y2 = rec.bbox
    w, h = x2 - x1, y2 - y1
    X1, X2 = max(0, int(x1 - mx * w)), min(img.shape[1], int(x2 + mx * w))
    Y1, Y2 = max(0, int(y1 - my * h)), min(img.shape[0], int(y2 + my * h))
    return img[Y1:Y2, X1:X2]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = {r.path.name: r for r in list_ccpd("ccpd_base") if r.blurriness >= 60 and r.bbox[2] - r.bbox[0] >= 220}
    sharp = [base[n] for n in stable_subset(sorted(base), 2, "demo")]
    for i, r in enumerate(sharp, 1):
        crop = plate_crop(r)
        imwrite(OUT / f"sharp_plate_{i}.png", crop)
        if i == 1:
            rng = np.random.default_rng(7)
            blurred = add_noise(convolve(crop, motion_psf(17, 20)), 0.01, rng)
            imwrite(OUT / "synthetic_motion_plate_L17_20deg.png", blurred)
    blur = {r.path.name: r for r in list_ccpd("ccpd_blur") if r.bbox[2] - r.bbox[0] >= 150}
    for i, n in enumerate(stable_subset(sorted(blur), 3, "demo"), 1):
        imwrite(OUT / f"ccpd_blur_real_{i}.png", plate_crop(blur[n]))
    page = render_text_page(5, (240, 480))
    imwrite(OUT / "sharp_text.png", np.repeat(page[..., None], 3, axis=-1))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
