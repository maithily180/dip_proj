"""Shared infrastructure for the experiment scripts.

* deterministic development / test splits of the sharp source images;
* the synthetic trial generator used by every sweep (identical grid for the
  baseline and for every variant, so results are directly comparable);
* a process-pool map with a progress bar;
* output locations and plotting defaults.
"""
from __future__ import annotations

import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from textdeblur.blur import add_noise, center_crop, convolve, crop_window  # noqa: E402
from textdeblur.data import (  # noqa: E402
    imread, kodak_paths, list_ccpd, render_text_page, stable_subset, to_gray,
)
from textdeblur.psf import motion_psf  # noqa: E402

RESULTS = ROOT / "results"
CSV = RESULTS / "csv"
FIG = RESULTS / "figures"
for _d in (CSV, FIG):
    _d.mkdir(parents=True, exist_ok=True)

WORKERS = int(os.environ.get("TEXTDEBLUR_WORKERS", max(1, (os.cpu_count() or 4) - 4)))

# ---------------------------------------------------------------------------
# sweep grid (identical for every estimator variant)
# ---------------------------------------------------------------------------
ANGLES = (0.0, 22.0, 45.0, 67.0, 90.0, 112.0, 135.0, 157.0)
LENGTHS = (5, 9, 13, 19, 27)
NOISES = (0.0, 0.01, 0.02, 0.04)
CROP = (96, 256)                  # plate-sized window used for all matched classes
MAX_LENGTH = 40                   # search range of the estimator in the sweeps

#: success criterion for a PSF estimate
ANGLE_TOL = 5.0
LENGTH_TOL = 2.0

CLASS_LABELS = {
    "natural_full": "Natural, full frame (512x768)",
    "natural": "Natural, plate-sized crop",
    "text": "Printed text, plate-sized crop",
    "plate": "License plate, plate-sized crop",
}


@dataclass(frozen=True)
class Source:
    """One sharp source image of the synthetic benchmark."""

    cls: str                # natural | text | plate
    name: str               # stable identifier
    split: str              # dev | test
    ref: str                # path, or page index for rendered text
    center: tuple[float, float] | None = None       # crop centre (plates)


@lru_cache(maxsize=1)
def sources() -> tuple[Source, ...]:
    """Deterministic list of sharp sources with their dev/test assignment."""
    out: list[Source] = []
    kod = kodak_paths()
    for i, p in enumerate(kod):
        out.append(Source("natural", p.stem, "dev" if i < 8 else "test", str(p)))
    for i in range(24):
        out.append(Source("text", f"page{i:02d}", "dev" if i < 8 else "test", str(i)))
    # sharp plates: CCPD's own sharpness score in the top range, typical size
    recs = [r for r in list_ccpd("ccpd_base")
            if r.blurriness >= 60 and 200 <= r.bbox[2] - r.bbox[0] <= 300]
    by_name = {r.path.name: r for r in recs}
    chosen = stable_subset(sorted(by_name), 32, salt="plates")
    for i, name in enumerate(chosen):
        r = by_name[name]
        x1, y1, x2, y2 = r.bbox
        out.append(Source("plate", f"plate{i:02d}", "dev" if i < 8 else "test",
                          str(r.path), ((y1 + y2) / 2, (x1 + x2) / 2)))
    return tuple(out)


@lru_cache(maxsize=64)
def load_source(cls: str, ref: str) -> np.ndarray:
    """Sharp grayscale source image in [0, 1]."""
    if cls == "text":
        return render_text_page(int(ref))
    return to_gray(imread(ref))


def seed_for(*parts) -> int:
    """Stable seed derived from the trial description."""
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def make_trial(src: Source, length: float, angle: float, sigma: float, full_frame: bool = False):
    """Blur a source, add noise and cut the evaluation window.

    The whole source is blurred *before* cropping, so the crop contains
    content smeared in from outside it, as a real photograph would.
    Returns ``(sharp_crop, blurred_crop)``.
    """
    sharp = load_source(src.cls, src.ref)
    rng = np.random.default_rng(seed_for(src.name, length, angle, sigma))
    blurred = add_noise(convolve(sharp, motion_psf(length, angle)), sigma, rng)
    if full_frame:
        return sharp, blurred
    if src.center is not None:
        cy, cx = src.center
        return crop_window(sharp, cy, cx, *CROP), crop_window(blurred, cy, cx, *CROP)
    return center_crop(sharp, *CROP), center_crop(blurred, *CROP)


def pmap(fn, jobs, workers: int | None = None, chunksize: int = 4, desc: str = ""):
    """Parallel map over ``jobs`` with a progress bar; preserves order."""
    from tqdm import tqdm

    jobs = list(jobs)
    workers = workers or WORKERS
    if workers <= 1:
        return [fn(j) for j in tqdm(jobs, desc=desc)]
    with ProcessPoolExecutor(workers) as ex:
        return list(tqdm(ex.map(fn, jobs, chunksize=chunksize), total=len(jobs), desc=desc))


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    })
    return plt
