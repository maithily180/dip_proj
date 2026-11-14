"""Dataset access: image I/O, CCPD annotations, Kodak, and printed-text pages.

All loaders return float images in ``[0, 1]`` (RGB channel order for colour).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "ROOT", "DATA", "imread", "imwrite", "to_gray", "CCPDRecord", "parse_ccpd_name",
    "list_ccpd", "kodak_paths", "render_text_page", "stable_subset",
]

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("TEXTDEBLUR_DATA", ROOT / "data"))

# --------------------------------------------------------------------------
# generic I/O
# --------------------------------------------------------------------------


def imread(path: str | os.PathLike, gray: bool = False) -> np.ndarray:
    """Read an image as float in [0, 1].  Works with non-ASCII Windows paths."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imdecode(buf, flag)
    if img is None:
        raise IOError(f"cannot read image: {path}")
    if not gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float64) / 255.0


def imwrite(path: str | os.PathLike, img: np.ndarray) -> None:
    """Write a float [0, 1] (or uint8) image; format chosen by extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = img
    if arr.dtype != np.uint8:
        arr = np.rint(np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(path.suffix, arr)
    if not ok:
        raise IOError(f"cannot encode image: {path}")
    buf.tofile(str(path))


def to_gray(img: np.ndarray) -> np.ndarray:
    """Luminance (ITU-R BT.601) of an RGB image; grayscale passes through."""
    img = np.asarray(img, dtype=float)
    if img.ndim == 2:
        return img
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


def stable_subset(names: list[str], n: int, salt: str = "") -> list[str]:
    """Deterministic pseudo-random subset: the ``n`` names with the smallest
    MD5 hash.  Independent of listing order and reproducible across machines."""
    key = lambda s: hashlib.md5((salt + s).encode("utf-8")).hexdigest()
    return sorted(sorted(names, key=key)[:n])


# --------------------------------------------------------------------------
# CCPD
# --------------------------------------------------------------------------

_PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
    "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警",
    "学", "O",
]
_ALPHABETS = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + ["O"]
_ADS = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") + ["O"]


@dataclass(frozen=True)
class CCPDRecord:
    """Annotation decoded from a CCPD file name."""

    path: Path
    bbox: tuple[int, int, int, int]            # x1, y1, x2, y2
    corners: tuple[tuple[int, int], ...]       # right-bottom, left-bottom, left-top, right-top
    plate: str                                  # full plate, e.g. "皖AY339S"
    brightness: int
    blurriness: int                             # CCPD's sharpness score (higher = sharper)

    @property
    def alnum(self) -> str:
        """The six alphanumeric characters (province glyph dropped)."""
        return self.plate[1:]


def parse_ccpd_name(path: str | os.PathLike) -> CCPDRecord:
    path = Path(path)
    parts = path.stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"not a CCPD file name: {path.name}")
    (x1, y1), (x2, y2) = [tuple(int(v) for v in p.split("&")) for p in parts[2].split("_")]
    corners = tuple(tuple(int(v) for v in p.split("&")) for p in parts[3].split("_"))
    idx = [int(v) for v in parts[4].split("_")]
    plate = _PROVINCES[idx[0]] + _ALPHABETS[idx[1]] + "".join(_ADS[i] for i in idx[2:])
    return CCPDRecord(path, (x1, y1, x2, y2), corners, plate, int(parts[5]), int(parts[6]))


def list_ccpd(split: str) -> list[CCPDRecord]:
    """All records of a CCPD split (``ccpd_base`` or ``ccpd_blur``), sorted."""
    folder = DATA / "ccpd" / split
    return [parse_ccpd_name(folder / f) for f in sorted(os.listdir(folder)) if f.endswith(".jpg")]


# --------------------------------------------------------------------------
# Kodak
# --------------------------------------------------------------------------


def kodak_paths() -> list[Path]:
    return sorted((DATA / "kodak").glob("kodim*.png"))


# --------------------------------------------------------------------------
# printed text
# --------------------------------------------------------------------------

_CORPUS = (
    "Motion blur is modelled as the convolution of a sharp image with a point spread "
    "function plus additive noise. Under linear uniform motion the kernel is a line "
    "segment described by two numbers, an angle and a length. The Fourier transform of "
    "that segment is a sinc whose zeros appear as dark parallel stripes in the spectrum "
    "of the blurred image. The Radon transform finds the orientation of the stripes and "
    "the cepstrum measures their spacing. Given the kernel, Richardson and Lucy showed "
    "how an iterative maximum likelihood update restores the latent image. PLATE 7429 KX "
    "ROAD CLOSED AHEAD 2026 EXIT 14B SPEED LIMIT 60 INVOICE NO 0058731 TOTAL DUE 1,249.50 "
    "The quick brown fox jumps over the lazy dog while 0123456789 vehicles pass gate "
    "number 35 at dawn. Registration AP09 CQ 4821 was recorded near the northern toll "
    "booth, and the operator noted that the characters were unreadable in frame 118."
)


def _font_candidates() -> list[str]:
    names = [
        "arial.ttf", "arialbd.ttf", "times.ttf", "timesbd.ttf", "cour.ttf", "courbd.ttf",
        "calibri.ttf", "verdana.ttf", "georgia.ttf", "tahoma.ttf", "consola.ttf",
        "DejaVuSans.ttf", "DejaVuSerif.ttf", "DejaVuSansMono.ttf", "LiberationSans-Regular.ttf",
    ]
    dirs = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
    ]
    found = []
    for d in dirs:
        for n in names:
            if (d / n).exists():
                found.append(str(d / n))
    return found


def render_text_page(index: int, shape: tuple[int, int] = (512, 768)) -> np.ndarray:
    """Render a deterministic printed-text page (grayscale, float [0, 1]).

    Page ``index`` selects the font, size, polarity and paper tone, so the set
    of pages covers the variety of printed text (body copy, bold signage-like
    text, light-on-dark).  Rendering uses fonts already present on the machine.
    """
    from PIL import Image, ImageDraw, ImageFont

    rng = np.random.default_rng(1000 + index)
    fonts = _font_candidates()
    h, w = shape
    size = int([15, 19, 24, 30, 38][index % 5])
    if fonts:
        font = ImageFont.truetype(fonts[index % len(fonts)], size)
    else:  # pragma: no cover - only when no TrueType font exists
        font = ImageFont.load_default()
    dark_on_light = (index % 4) != 3
    paper = float(rng.uniform(0.78, 0.95)) if dark_on_light else float(rng.uniform(0.08, 0.25))
    ink = float(rng.uniform(0.05, 0.25)) if dark_on_light else float(rng.uniform(0.80, 0.95))

    img = Image.new("L", (w, h), int(paper * 255))
    draw = ImageDraw.Draw(img)
    words = _CORPUS.split()
    start = int(rng.integers(0, len(words)))
    words = words[start:] + words[:start]
    y, line_h, margin, k = 6, int(size * 1.35), 8, 0
    while y < h - line_h:
        line = ""
        while k < len(words) * 6:
            trial = (line + " " + words[k % len(words)]).strip()
            if draw.textlength(trial, font=font) > w - 2 * margin:
                break
            line, k = trial, k + 1
        draw.text((margin, y), line, fill=int(ink * 255), font=font)
        y += line_h
    page = np.asarray(img, dtype=np.float64) / 255.0
    # mild paper shading + grain so the background is not perfectly flat
    yy, xx = np.mgrid[:h, :w]
    shade = 1.0 + 0.05 * np.sin(xx / w * np.pi * rng.uniform(0.5, 1.5) + rng.uniform(0, 3))
    page = np.clip(page * shade + rng.normal(0, 0.004, page.shape), 0, 1)
    return page
