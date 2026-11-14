"""OCR as a measuring instrument.

OCR is **not** part of the deblurring algorithm.  It is used only to answer
"did the restoration make the text more readable?" on data that has no sharp
ground truth.  Removing this module leaves the pipeline untouched.

The reader is the PP-OCR text-recognition network run through ONNX Runtime
(``rapidocr-onnxruntime``).  Plates are first rectified to a fronto-parallel
strip using the four annotated corners, so that recognition - not detection -
is what is being measured.
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["PlateReader", "rectify_plate", "normalize_plate_text"]


def rectify_plate(
    img: np.ndarray,
    corners: tuple[tuple[float, float], ...],
    offset: tuple[float, float] = (0.0, 0.0),
    out_hw: tuple[int, int] = (96, 300),
    margin: float = 0.08,
) -> np.ndarray:
    """Warp a plate to a fixed-size fronto-parallel strip.

    ``corners`` are (x, y) pairs in CCPD order: right-bottom, left-bottom,
    left-top, right-top, in the coordinates of the full image; ``offset`` is
    the top-left corner of ``img`` inside that image when ``img`` is a crop.
    Returns a float RGB (or gray) image in [0, 1].
    """
    off = np.asarray(offset, dtype=float)
    rb, lb, lt, rt = [np.asarray(c, dtype=float) - off for c in corners]
    h, w = out_hw
    mx, my = margin * w, margin * h
    src = np.array([lt, rt, rb, lb], dtype=np.float32)
    dst = np.array([[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(
        np.clip(img, 0, 1).astype(np.float32), m, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return np.clip(out, 0, 1).astype(float)


def normalize_plate_text(text: str, n_chars: int = 6) -> str:
    """Keep upper-case ASCII letters/digits and return the last ``n_chars``.

    Chinese plates are one province glyph followed by six alphanumerics; the
    glyph is dropped so that the score reflects stroke legibility of Latin
    characters and digits only.
    """
    s = "".join(ch for ch in text.upper() if ch.isascii() and ch.isalnum())
    return s[-n_chars:] if len(s) > n_chars else s


class PlateReader:
    """Thin wrapper around the recognition model (lazy-loaded)."""

    def __init__(self) -> None:
        self._engine = None

    def _load(self):
        if self._engine is None:
            import logging

            logging.getLogger("RapidOCR").setLevel(logging.ERROR)
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def read(self, img: np.ndarray) -> tuple[str, float]:
        """Recognise a single text line.  Returns ``(text, confidence)``."""
        arr = np.rint(np.clip(img, 0, 1) * 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        res, _ = self._load()(arr, use_det=False, use_cls=False, use_rec=True)
        if not res:
            return "", 0.0
        return "".join(r[0] for r in res), float(np.mean([r[1] for r in res]))

    def read_plate(self, img: np.ndarray) -> str:
        """Six alphanumeric plate characters read from a rectified plate."""
        return normalize_plate_text(self.read(img)[0])
