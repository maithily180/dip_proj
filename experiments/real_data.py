"""Loading and scoring real license-plate images (shared by experiments 5 and 7).

Two public datasets are used:

* **LPBlur** (Gong et al., 2024) - real motion-blurred plates photographed
  from moving vehicles, each paired with a sharp exposure of the same plate.
  It has no text labels, so the text read from the *sharp* image is used as
  the reference; pairs whose sharp image does not yield six characters are
  excluded.  Sharp pairs also allow PSNR/SSIM on real blur.
* **CCPD-Blur** (Xu et al., 2018) - real blurred plates in full street
  photographs.  The plate text is encoded in the file name; there is no sharp
  counterpart.  The plate region (bounding box plus margin) is restored and
  then rectified with the annotated corners before recognition.

OCR is used purely as a read-out; it is never part of the restoration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from common import ROOT  # noqa: F401  (adds the repo to sys.path)
from textdeblur.data import DATA, imread, list_ccpd, parse_ccpd_name, stable_subset
from textdeblur.ocr import PlateReader, rectify_plate

LPBLUR = DATA / "lpblur" / "LPBlur"
_reader = PlateReader()


@dataclass(frozen=True)
class RealItem:
    dataset: str          # lpblur | ccpd
    name: str
    path: str


def lpblur_items(n: int, split: str) -> list[RealItem]:
    """Deterministic LPBlur subset.  ``dev`` and ``test`` never overlap."""
    names = sorted(os.listdir(LPBLUR / "blur"))
    dev = set(stable_subset(names, 150, "dev"))
    if split == "dev":
        chosen = sorted(dev)[:n]
    else:
        chosen = stable_subset([f for f in names if f not in dev], n, "test")
    return [RealItem("lpblur", f, str(LPBLUR / "blur" / f)) for f in chosen]


def ccpd_items(n: int, split: str) -> list[RealItem]:
    """Deterministic CCPD-Blur subset.  ``dev`` and ``test`` never overlap."""
    paths = [str(r.path) for r in _ccpd_blur()]
    dev = set(stable_subset(paths, 150, "dev"))
    if split == "dev":
        chosen = sorted(dev)[:n]
    else:
        chosen = stable_subset([p for p in paths if p not in dev], n, "test")
    return [RealItem("ccpd", Path(p).name, p) for p in chosen]


@lru_cache(maxsize=1)
def _ccpd_blur():
    return list_ccpd("ccpd_blur")


def plate_region(img: np.ndarray, bbox, mx: float = 0.25, my: float = 0.6):
    """Plate bounding box grown by a margin; returns (crop, (x0, y0))."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    X1, X2 = max(0, int(x1 - mx * w)), min(img.shape[1], int(x2 + mx * w))
    Y1, Y2 = max(0, int(y1 - my * h)), min(img.shape[0], int(y2 + my * h))
    return img[Y1:Y2, X1:X2], (X1, Y1)


@dataclass
class Loaded:
    """A real item ready for the pipeline."""

    blurred: np.ndarray                 # region handed to the estimator / restorer
    truth: str                          # reference plate text (6 alphanumerics)
    sharp: np.ndarray | None            # paired sharp image (LPBlur only)
    reader: callable                    # image -> recognised plate text


def load_item(item: RealItem) -> Loaded:
    if item.dataset == "lpblur":
        blurred = imread(item.path)
        sharp = imread(str(LPBLUR / "sharp" / item.name))
        truth = _reader.read_plate(sharp)
        return Loaded(blurred, truth, sharp, _reader.read_plate)
    rec = parse_ccpd_name(item.path)
    region, off = plate_region(imread(item.path), rec.bbox)
    read = lambda im: _reader.read_plate(rectify_plate(im, rec.corners, off))
    return Loaded(region, rec.alnum, None, read)
