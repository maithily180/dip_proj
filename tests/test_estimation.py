"""Tests for the frequency-domain estimator, the gate and the full pipeline."""
import numpy as np
import pytest

from textdeblur.blur import add_noise, center_crop, convolve
from textdeblur.cepstrum import cepstrum2d, polar_cepstrum, radon_cepstrum, weighted_log_spectrum
from textdeblur.data import render_text_page
from textdeblur.estimate import (
    BASELINE, TEXT_AWARE, EstimatorConfig, clip_block_grid, estimate_blur, radon_transform,
)
from textdeblur.gate import GateConfig, auc, calibrate_threshold, confusion, decide, roc_curve
from textdeblur.metrics import angle_error, psnr
from textdeblur.pipeline import BASELINE_PIPELINE, PipelineConfig, deblur, deblur_with_psf
from textdeblur.preprocess import PREPROCESSORS, preprocess
from textdeblur.psf import motion_psf
from textdeblur.radon_angle import estimate_two_stage
from textdeblur.spectrum import log_spectrum, prepare


def make(length, angle, sigma=0.0, page=2, hw=(96, 256), seed=0):
    sharp = render_text_page(page, (200, 400))
    blurred = add_noise(convolve(sharp, motion_psf(length, angle)), sigma, np.random.default_rng(seed))
    return center_crop(sharp, *hw), center_crop(blurred, *hw)


# --------------------------------------------------------- spectrum -------

def test_prepare_pads_to_square_and_removes_mean():
    x = prepare(np.random.default_rng(0).random((40, 100)))
    assert x.shape == (128, 128)
    assert abs(x.sum()) < 1e-9
    assert log_spectrum(np.random.default_rng(0).random((40, 100))).shape == (128, 128)


def test_cepstrum_has_dip_at_blur_length():
    _, b = make(15, 0, hw=(128, 256))
    c = cepstrum2d(b)
    n = c.shape[0]
    row = c[n // 2, n // 2 + 5 : n // 2 + 40]
    assert int(np.argmin(row)) + 5 == 15


def test_projection_slice_equivalence():
    """Radon -> 1-D FFT gives the same profiles as slicing the 2-D cepstrum."""
    _, b = make(13, 35)
    angles = np.arange(0, 180, 3.0)
    polar = polar_cepstrum(cepstrum2d(b), angles, 40)[:, 4:]
    rad = radon_cepstrum(weighted_log_spectrum(b), angles, 40)[:, 4:]
    assert np.corrcoef(polar.ravel(), rad.ravel())[0, 1] > 0.97


def test_radon_transform_matches_skimage():
    import warnings
    from skimage.transform import radon

    _, b = make(13, 35)
    spec = weighted_log_spectrum(b)
    angles = np.arange(0, 180, 7.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = radon(spec, theta=angles, circle=True)
    ours = radon_transform(spec, angles)
    assert np.abs(ours - ref).max() < 1e-3 * np.abs(ref).max()


# -------------------------------------------------------- estimator -------

@pytest.mark.parametrize("length,angle", [(7, 0), (9, 30), (13, 75), (17, 90), (21, 120), (27, 160)])
def test_estimator_recovers_clean_blur(length, angle):
    _, b = make(length, angle)
    est = estimate_blur(b, BASELINE.with_(max_length=40))
    assert angle_error(est.angle, angle) <= 3.0
    assert abs(est.length - length) <= 1.5
    assert est.confidence > 5


def test_estimator_handles_mild_noise_and_colour():
    _, b = make(15, 40, sigma=0.01)
    rgb = np.stack([b, b * 0.9, b * 0.7], axis=-1)
    est = estimate_blur(rgb, BASELINE.with_(max_length=40))
    assert angle_error(est.angle, 40) <= 3.0 and abs(est.length - 15) <= 1.5
    assert est.psf().sum() == pytest.approx(1.0)
    assert est.profile.shape == est.lags.shape


def test_estimator_options_run():
    _, b = make(11, 20, sigma=0.01)
    for cfg in (
        EstimatorConfig(cutoff="auto"), EstimatorConfig(notch_grid=True),
        EstimatorConfig(angular_detrend=True), EstimatorConfig(window="hann", refine=False), TEXT_AWARE,
    ):
        est = estimate_blur(b, cfg.with_(max_length=40))
        assert 0 <= est.angle < 180 and est.length >= 4


def test_confidence_separates_blurred_from_sharp():
    sharp, b = make(15, 30)
    cfg = BASELINE.with_(max_length=40)
    assert estimate_blur(b, cfg).confidence > 2 * estimate_blur(sharp, cfg).confidence


def test_two_stage_reference_runs():
    _, b = make(15, 0)
    est = estimate_two_stage(b, EstimatorConfig(max_length=40))
    assert 0 <= est.angle < 180 and est.length >= 4


def test_block_grid_clip_only_lowers_values():
    spec = np.random.default_rng(0).random((64, 64))
    spec[:, 32 + 8] += 5.0                                   # a bright grid line
    out = clip_block_grid(spec)
    assert np.all(out <= spec + 1e-12)
    assert out[:, 40].max() < 2.0


@pytest.mark.parametrize("name", sorted(PREPROCESSORS))
def test_preprocessors_return_2d_float(name):
    rgb = np.random.default_rng(0).random((48, 96, 3))
    out = preprocess(rgb, name)
    assert out.shape == (48, 96) and np.isfinite(out).all()


def test_unknown_preprocessor_raises():
    with pytest.raises(ValueError):
        preprocess(np.zeros((8, 8)), "nope")


# ------------------------------------------------------------- gate -------

def test_gate_decisions():
    _, b = make(15, 30)
    est = estimate_blur(b, BASELINE.with_(max_length=40))
    assert decide(est, GateConfig(min_confidence=1.0)).accept
    refused = decide(est, GateConfig(min_confidence=1e9))
    assert not refused.accept and "weak" in refused.reason
    assert not decide(est, GateConfig(min_peak_ratio=1e9)).accept


def test_roc_and_calibration():
    rng = np.random.default_rng(0)
    good = np.r_[np.ones(500, bool), np.zeros(500, bool)]
    scores = np.r_[rng.normal(3, 1, 500), rng.normal(0, 1, 500)]
    fpr, tpr, thr = roc_curve(scores, good)
    assert fpr[0] == 0 and tpr[0] == 0 and fpr[-1] == 1 and tpr[-1] == 1
    assert 0.95 < auc(fpr, tpr) < 1.0
    t = calibrate_threshold(scores, good, max_false_accept=0.05)
    accept = scores >= t
    assert np.mean(accept[~good]) <= 0.05
    cm = confusion(accept, good)
    assert sum(cm.values()) == 1000
    assert cm["correct_restore"] > cm["missed_restore"]


# --------------------------------------------------------- pipeline -------

def test_pipeline_end_to_end_improves_image():
    sharp, b = make(15, 30, sigma=0.005)
    res = deblur(b, BASELINE_PIPELINE.with_(estimator=BASELINE.with_(max_length=40), num_iter=40))
    assert res.accepted
    assert psnr(res.restored, sharp, 8) > psnr(b, sharp, 8) + 1.5
    assert res.restored.min() >= 0 and res.restored.max() <= 1


def test_gate_refusal_returns_input_unchanged():
    _, b = make(15, 30)
    cfg = PipelineConfig(estimator=BASELINE.with_(max_length=40), gate=GateConfig(min_confidence=1e9))
    res = deblur(b, cfg)
    assert not res.accepted
    assert np.array_equal(res.restored, b)
    assert res.estimate is not None


def test_semi_blind_and_wiener_paths():
    sharp, b = make(13, 60)
    res = deblur_with_psf(b, 13, 60, PipelineConfig(num_iter=40))
    assert res.accepted and psnr(res.restored, sharp, 8) > psnr(b, sharp, 8) + 1.5
    w = deblur_with_psf(b, 13, 60, PipelineConfig(method="wiener"))
    assert psnr(w.restored, sharp, 8) > psnr(b, sharp, 8)
    with pytest.raises(ValueError):
        deblur_with_psf(b, 13, 60, PipelineConfig(method="magic"))
