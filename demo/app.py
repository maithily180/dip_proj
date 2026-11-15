"""Interactive demo: blurred input | estimated PSF + confidence | restored output.

Run with::

    streamlit run demo/app.py

Modes
-----
* Automatic  - the PSF is estimated from the image; the confidence gate may
  refuse to deconvolve.
* Semi-blind - you set the blur angle and length yourself (for images where
  automatic estimation is not confident) and only the non-blind step runs.
* Simulate   - blur a sharp image with a known kernel, then watch the pipeline
  recover it; the true and estimated parameters are shown side by side.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from textdeblur.blur import add_noise, convolve  # noqa: E402
from textdeblur.data import imread, render_text_page, to_gray  # noqa: E402
from textdeblur.estimate import BASELINE, TEXT_AWARE, estimate_blur  # noqa: E402
from textdeblur.gate import DEFAULT_GATE, GateConfig, decide  # noqa: E402
from textdeblur.metrics import laplacian_variance, psnr  # noqa: E402
from textdeblur.pipeline import PipelineConfig, restore  # noqa: E402
from textdeblur.psf import motion_psf  # noqa: E402

SAMPLES = ROOT / "demo" / "samples"

st.set_page_config(page_title="Seeing Through the Shake", layout="wide")


# ----------------------------------------------------------------- helpers --

def load_upload(file) -> np.ndarray:
    import cv2

    buf = np.frombuffer(file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    scale = 900.0 / max(img.shape[:2])
    if scale < 1:                                   # keep the demo responsive
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def psf_figure(psf: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    ax.imshow(psf, cmap="magma", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def analysis_figure(est, true=None):
    """Log-spectrum, Radon-cepstrum map and the profile along the estimate."""
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    spec = est.spectrum
    ax[0].imshow(spec, cmap="gray", vmin=np.percentile(spec, 2), vmax=np.percentile(spec, 99.5))
    ax[0].set_title("log-magnitude spectrum\n(dark stripes are the PSF nulls, perpendicular to motion)")
    ax[0].set_xticks([]); ax[0].set_yticks([])

    lo = int(np.ceil(est.config.min_length))
    prof = est.profiles
    v = np.percentile(np.abs(prof[:, lo:]), 99.5)
    ax[1].imshow(prof[:, lo:], aspect="auto", cmap="RdBu", vmin=-v, vmax=v,
                 extent=[lo, prof.shape[1] - 1, 180, 0])
    ax[1].plot(est.length, est.angle, "o", mfc="none", mec="k", ms=14, mew=1.8, label="detected dip")
    if true is not None:
        ax[1].plot(true[1], true[0], "x", color="lime", ms=10, mew=2, label="true PSF")
    ax[1].set_xlabel("lag = candidate blur length (px)"); ax[1].set_ylabel("Radon projection angle (deg)")
    ax[1].set_title("Radon-cepstrum map\n(1-D spectrum of every Radon projection)")
    ax[1].legend(fontsize=8, loc="upper right")

    p = est.profile
    ax[2].plot(est.lags[lo:], p[lo:], color="#1f77b4")
    ax[2].axvline(est.length, color="k", ls="--", lw=1, label=f"estimated length {est.length:.1f} px")
    if true is not None:
        ax[2].axvline(true[1], color="green", ls=":", lw=1.5, label=f"true length {true[1]:g} px")
    ax[2].set_xlabel("lag (px)"); ax[2].set_ylabel("cepstrum")
    ax[2].set_title(f"cepstral profile along {est.angle:.1f} deg")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def sample_list() -> dict[str, str]:
    out = {}
    if SAMPLES.exists():
        for p in sorted(SAMPLES.glob("*.png")) + sorted(SAMPLES.glob("*.jpg")):
            out[p.stem.replace("_", " ")] = str(p)
    return out


# ------------------------------------------------------------------ layout --

st.title("Seeing Through the Shake")
st.caption("Text-aware PSF estimation for blind motion deblurring - Radon transform, cepstrum, "
           "Richardson-Lucy. No training data, no neural network in the pipeline.")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Mode", ["Automatic (blind)", "Semi-blind (manual PSF)", "Simulate blur on a sharp image"])
    upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
    samples = sample_list()
    choice = st.selectbox("...or pick a sample", list(samples) or ["(no samples found)"])

    st.header("Estimator")
    variant = st.radio("PSF estimator", ["Text-aware (pre-processing on)", "Baseline (plain grayscale)"])
    st.header("Confidence gate")
    use_gate = st.checkbox("Refuse when the estimate is unreliable", value=True)
    thr = st.slider("Minimum confidence (robust sigma)", 0.0, 30.0, float(DEFAULT_GATE.min_confidence), 0.5)
    ratio = st.slider("Minimum peak ratio", 0.0, 3.0, float(DEFAULT_GATE.min_peak_ratio), 0.05)

    st.header("Restoration")
    method = st.radio("Deconvolution", ["Richardson-Lucy", "Wiener"])
    iters = st.slider("RL iterations", 5, 200, 40, 5)
    nsr = st.select_slider("Wiener noise-to-signal ratio", [0.001, 0.003, 0.01, 0.03, 0.1], 0.01)

# ------------------------------------------------------------------- input --

if upload is not None:
    img = load_upload(upload)
elif samples:
    img = imread(samples[choice])
else:
    img = np.repeat(render_text_page(2, (240, 480))[..., None], 3, axis=-1)

true = None
sharp = None
if mode.startswith("Simulate"):
    c1, c2, c3 = st.columns(3)
    t_len = c1.slider("True blur length (px)", 3, 40, 15)
    t_ang = c2.slider("True blur angle (deg)", 0, 179, 30)
    t_sig = c3.slider("Noise sigma (x 1/255)", 0.0, 12.0, 2.0, 0.5) / 255.0
    sharp = img
    img = add_noise(convolve(sharp, motion_psf(t_len, t_ang)), t_sig, np.random.default_rng(0))
    true = (float(t_ang), float(t_len))

est_cfg = TEXT_AWARE if variant.startswith("Text") else BASELINE
pipe = PipelineConfig(estimator=est_cfg, method="rl" if method.startswith("Rich") else "wiener",
                      num_iter=iters, nsr=nsr)

# ---------------------------------------------------------------- pipeline --

est = estimate_blur(img, est_cfg)
decision = decide(est, GateConfig(thr, ratio)) if use_gate else None

if mode.startswith("Semi"):
    c1, c2 = st.columns(2)
    m_len = c1.slider("Blur length (px)", 2.0, 80.0, float(np.clip(est.length, 2, 80)), 0.5)
    m_ang = c2.slider("Blur angle (deg, counter-clockwise from horizontal)", 0.0, 179.5,
                      float(est.angle), 0.5)
    psf = motion_psf(m_len, m_ang)
    restored, accepted = restore(img, psf, pipe), True
    verdict = f"Semi-blind: using the PSF you set (length {m_len:.1f} px, angle {m_ang:.1f} deg)."
else:
    psf = est.psf()
    accepted = decision is None or decision.accept
    restored = restore(img, psf, pipe) if accepted else img
    verdict = decision.reason if decision is not None else "gate disabled - always deconvolving"

# ------------------------------------------------------------------ output --

col1, col2, col3 = st.columns([5, 3, 5])
with col1:
    st.subheader("Blurred input")
    st.image(np.clip(img, 0, 1), use_container_width=True)
with col2:
    st.subheader("Estimated PSF")
    st.pyplot(psf_figure(psf, f"length {est.length:.1f} px, angle {est.angle:.1f} deg"
                         if not mode.startswith("Semi") else "user-supplied PSF"))
    st.metric("Confidence", f"{est.confidence:.1f} sigma")
    st.metric("Peak ratio", f"{est.peak_ratio:.2f}")
    if true is not None:
        st.caption(f"True PSF: length {true[1]:g} px, angle {true[0]:g} deg")
with col3:
    st.subheader("Restored output" if accepted else "Output (restoration withheld)")
    st.image(np.clip(restored, 0, 1), use_container_width=True)

if accepted:
    st.success(f"Restored. {verdict}")
else:
    st.warning(f"Gate refused to deconvolve: {verdict}. The input is returned unchanged - "
               f"a wrong PSF would make the image worse, not better. Try the semi-blind mode.")

m1, m2, m3 = st.columns(3)
m1.metric("Sharpness (variance of Laplacian), input", f"{laplacian_variance(img) * 1e4:.2f}")
m2.metric("Sharpness, output", f"{laplacian_variance(restored) * 1e4:.2f}")
if sharp is not None:
    m3.metric("PSNR vs. sharp original", f"{psnr(restored, sharp, 8):.2f} dB",
              f"{psnr(restored, sharp, 8) - psnr(img, sharp, 8):+.2f} dB vs. blurred")

with st.expander("How the estimate was made (frequency-domain evidence)", expanded=True):
    st.pyplot(analysis_figure(est, true))
    st.markdown(
        "A linear motion blur multiplies the image spectrum by a sinc whose zeros are parallel dark "
        "stripes perpendicular to the motion, spaced 1/L apart. The Radon transform projects the "
        "log-spectrum along every direction; the 1-D Fourier transform of each projection is a radial "
        "slice of the cepstrum. The direction and lag of the deepest negative dip give the blur angle "
        "and length, and how far that dip stands out from the rest of the map is the confidence."
    )
