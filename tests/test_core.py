"""Unit tests for the PSF model, degradation, deconvolution and metrics."""
import numpy as np
import pytest

from textdeblur.blur import add_noise, center_crop, convolve, crop_window, degrade
from textdeblur.data import parse_ccpd_name, render_text_page, stable_subset, to_gray
from textdeblur.deconv import blur_fft, richardson_lucy, wiener
from textdeblur.metrics import (
    angle_error, char_accuracy, edit_distance, kernel_similarity, laplacian_variance, psnr, ssim,
)
from textdeblur.psf import angle_diff, motion_psf, pad_psf_to


# ---------------------------------------------------------------- PSF ------

@pytest.mark.parametrize("length", [3, 5, 9, 15, 21])
def test_horizontal_psf_is_exact_box(length):
    k = motion_psf(length, 0)
    row = k[k.shape[0] // 2]
    assert np.isclose(k.sum(), 1.0)
    assert np.count_nonzero(row > 1e-9) == length
    assert np.allclose(row[row > 1e-9], 1.0 / length, atol=1e-6)
    assert np.allclose(np.delete(k, k.shape[0] // 2, axis=0), 0.0)


@pytest.mark.parametrize("angle", [0, 17, 45, 90, 133, 179])
def test_psf_normalised_and_symmetric(angle):
    k = motion_psf(13, angle)
    assert np.isclose(k.sum(), 1.0)
    assert k.min() >= 0
    assert np.allclose(k, k[::-1, ::-1], atol=1e-3)     # a line through the centre


def test_angle_convention_is_counter_clockwise():
    k = motion_psf(21, 45)
    c = k.shape[0] // 2
    upper_right = k[:c, c + 1 :].sum()        # rows above centre, columns right of it
    upper_left = k[:c, :c].sum()
    assert upper_right > 10 * upper_left
    assert np.allclose(motion_psf(15, 90), motion_psf(15, 0).T, atol=1e-6)


def test_psf_period_180_and_delta():
    assert np.allclose(motion_psf(11, 30), motion_psf(11, 210))
    d = motion_psf(1, 77)
    assert d.sum() == 1.0 and d[d.shape[0] // 2, d.shape[1] // 2] == 1.0


def test_angle_diff_wraps():
    assert angle_diff(2, 178) == 4
    assert angle_diff(90, 0) == 90
    assert angle_error(181, 1) == 0


def test_pad_psf_matches_spatial_convolution():
    rng = np.random.default_rng(0)
    img = rng.random((40, 56))
    k = motion_psf(7, 30)
    from scipy import ndimage
    ref = ndimage.convolve(img, k, mode="wrap")
    assert np.allclose(blur_fft(img, k), ref, atol=1e-10)
    assert np.isclose(pad_psf_to(k, (40, 56)).sum(), 1.0)


# ------------------------------------------------------------- blur -------

def test_degrade_preserves_mean_and_shape():
    page = render_text_page(0, (128, 256))
    b, k = degrade(page, 11, 20, noise_sigma=0.0)
    assert b.shape == page.shape
    assert abs(b.mean() - page.mean()) < 5e-3
    assert np.isclose(k.sum(), 1.0)


def test_noise_is_seeded_and_quantised():
    x = np.full((32, 32), 0.5)
    a = add_noise(x, 0.05, np.random.default_rng(1))
    b = add_noise(x, 0.05, np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert np.allclose(a * 255, np.rint(a * 255))
    assert 0.03 < a.std() < 0.07


def test_convolve_colour_and_crops():
    rgb = np.random.default_rng(0).random((40, 60, 3))
    out = convolve(rgb, motion_psf(5, 0))
    assert out.shape == rgb.shape
    assert center_crop(rgb, 20, 30).shape == (20, 30, 3)
    assert crop_window(rgb, 0, 0, 20, 30).shape == (20, 30, 3)      # clamped to the border
    with pytest.raises(ValueError):
        crop_window(rgb, 5, 5, 100, 30)


# ----------------------------------------------------------- deconv -------

@pytest.fixture(scope="module")
def blurred_page():
    page = render_text_page(1, (160, 320))
    k = motion_psf(13, 25)
    b = add_noise(convolve(page, k), 0.005, np.random.default_rng(0))
    return center_crop(page, 96, 256), center_crop(b, 96, 256), k


def test_rl_with_true_psf_improves_psnr(blurred_page):
    sharp, blurred, k = blurred_page
    restored = richardson_lucy(blurred, k, num_iter=40)
    assert psnr(restored, sharp) > psnr(blurred, sharp) + 2.0
    assert ssim(restored, sharp) > ssim(blurred, sharp)


def test_rl_has_no_border_artifacts(blurred_page):
    sharp, blurred, k = blurred_page
    restored = richardson_lucy(blurred, k, num_iter=40)
    # the error in the outer band must not blow up relative to the interior
    err = (np.clip(restored, 0, 1) - sharp) ** 2
    inner = err[16:-16, 16:-16].mean()
    outer = (err.sum() - err[16:-16, 16:-16].sum()) / (err.size - err[16:-16, 16:-16].size)
    assert outer < 3 * inner


def test_rl_is_nonnegative_and_handles_colour(blurred_page):
    _, blurred, k = blurred_page
    rgb = np.stack([blurred, blurred * 0.8, blurred * 0.6], axis=-1)
    out = richardson_lucy(rgb, k, num_iter=5)
    assert out.shape == rgb.shape and out.min() >= 0


def test_wiener_with_true_psf_improves_psnr(blurred_page):
    sharp, blurred, k = blurred_page
    restored = wiener(blurred, k, nsr=0.01)
    assert psnr(restored, sharp) > psnr(blurred, sharp) + 1.0


def test_wrong_psf_is_worse_than_right_psf(blurred_page):
    sharp, blurred, k = blurred_page
    good = richardson_lucy(blurred, k, num_iter=40)
    bad = richardson_lucy(blurred, motion_psf(25, 115), num_iter=40)
    assert psnr(bad, sharp) < psnr(good, sharp) - 3.0
    assert psnr(bad, sharp) < psnr(blurred, sharp)        # worse than doing nothing


# ---------------------------------------------------------- metrics -------

def test_text_metrics():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert char_accuracy("A1234B", "A1234B") == 1.0
    assert np.isclose(char_accuracy("A1234", "A1234B"), 5 / 6)
    assert char_accuracy("ZZZZZZZZZZZZ", "A1234B") == 0.0


def test_image_metrics():
    a = np.random.default_rng(0).random((32, 32))
    assert psnr(a, a) == float("inf")
    assert np.isclose(psnr(np.zeros((8, 8)), np.full((8, 8), 0.1)), 20.0)
    assert np.isclose(ssim(a, a), 1.0)
    from scipy import ndimage
    assert laplacian_variance(a) > laplacian_variance(ndimage.gaussian_filter(a, 2))


def test_kernel_similarity():
    k = motion_psf(11, 40)
    assert np.isclose(kernel_similarity(k, k), 1.0)
    assert kernel_similarity(k, motion_psf(11, 130)) < 0.5
    shifted = np.roll(np.pad(k, 3), 2, axis=1)
    assert kernel_similarity(k, shifted) > 0.99


# ------------------------------------------------------------- data -------

def test_parse_ccpd_name():
    name = "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
    r = parse_ccpd_name(name)
    assert r.bbox == (154, 383, 386, 473)
    assert r.corners[2] == (154, 383)
    assert r.plate[1:] == "AY339S" and r.alnum == "AY339S"
    assert (r.brightness, r.blurriness) == (37, 15)
    with pytest.raises(ValueError):
        parse_ccpd_name("not-a-ccpd-name.jpg")


def test_text_page_and_subset_are_deterministic():
    a, b = render_text_page(3, (120, 200)), render_text_page(3, (120, 200))
    assert a.shape == (120, 200) and np.array_equal(a, b)
    assert a.std() > 0.05                                   # there is ink on the page
    names = [f"f{i}" for i in range(50)]
    assert stable_subset(names, 7) == stable_subset(names[::-1], 7)
    assert to_gray(np.ones((4, 4, 3))).shape == (4, 4)
