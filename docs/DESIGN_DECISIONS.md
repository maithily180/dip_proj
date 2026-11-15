# Design decisions

What was tried, what was kept, and why. Every choice below was made on **development data**
(synthetic dev split, LPBlur dev subset, CCPD-Blur dev subset). The reported results in the report
come from test images that none of these decisions saw. `experiments/00_design_dev.py`
regenerates the development evidence for decisions 3 and 4; `experiments/04_gate_calibration.py`
regenerates decision 5.

## 1. Joint (angle, length) search instead of the two-stage textbook rule

*Phase 0* found that picking the angle as the direction of maximum Radon-projection variance is
fragile: the image's own dominant edges (character strokes, plate borders) produce high-variance
projections too, and the angle is decided before any cepstral evidence is consulted. A joint search
over the full (direction × lag) cepstrum map was more reliable.

Kept: the deepest negative dip of the Radon-cepstrum map gives both parameters. The angle is
still Radon-based and the length cepstrum-based, so this stays within the proposal's pipeline and does
not turn into a Radon-vs-cepstrum benchmark (out of scope). The two-stage rule stays in the code
(`textdeblur/radon_angle.py`) as a reference row in experiment 2.

## 2. Real Radon transform, isotropic grid, fast implementation

Sampling the cepstrum on a polar grid was worse than a real Radon transform at short lengths
(interpolation blurs the narrow dip). `skimage.transform.radon` was too slow for 180 directions ×
tens of thousands of trials, so `radon_transform` rotates with OpenCV and sums columns. It matches
skimage to numerical precision (unit test) and is ~10× faster. Non-square crops are zero-padded to a
square FFT grid of at least 256 so that every direction has the same frequency resolution.

## 3. The "text-aware" stage: band limiting, not edge enhancement

The proposal planned local contrast normalisation + edge enhancement in front of the estimator.
We implemented nine operators (CLAHE, unsharp mask, Gaussian high-pass, Laplacian, DoG, gradient
magnitude, LCN, homomorphic filtering, LCN + Laplacian) and tested them on the synthetic dev
sweep and on real plates.

* On synthetic data none of them improved estimation success; edge emphasis lowered it under
  noise, because it amplifies the high-frequency band where noise dominates and where the sinc
  nulls are already the weakest.
* On real LPBlur plates (dev), contrast and edge operators lowered OCR accuracy after restoration,
  and LCN collapsed it.

The analysis of the failures pointed the opposite way. Real plate crops are small, upsampled and
JPEG-compressed: above some radius their spectrum carries no genuine image signal, only noise and
compression residue that bury the blur nulls. Restricting the spectrum to a lower band:

| LPBlur dev (n=147), OCR char. accuracy after RL-30 | |
|---|---|
| blurred input | 0.234 |
| full band (classical) | 0.359 |
| SNR-adaptive band limit | 0.490 |
| band limit 0.5 | 0.518 |
| band limit 0.3 | 0.587 |
| multi-band search {1, 0.5, 0.3} | 0.545 |

(first development run; `00_design_dev.py` reproduces the comparison with the final code.)

A fixed limit of 0.3 Nyquist is the best on LPBlur but **destroys short blurs**: the first null of an
L-pixel blur lies at 2/L of Nyquist, so for L ≤ 9 there is no null inside the band (synthetic dev
success at L = 5 drops from 64% to 0%). The multi-band search runs the joint search at limits
{1, 0.5, 0.3} and keeps the estimate whose dip is most *directional* (angular z-score). Synthetic dev:
84.0% success vs. 85.6% for the full band (L = 5: 54% vs. 64%); LPBlur dev: 0.545 vs. 0.359.
Selecting by dip depth instead of directionality picked the full band too often (0.359); also
selecting by a relaxed rule favouring low bands gave at most +0.05 on LPBlur at a further synthetic
cost, so the plain argmax was kept.

**Final:** `TEXT_AWARE = EstimatorConfig(cutoff=(1.0, 0.5, 0.3))`.

Also tried and rejected: clipping the JPEG 8×8 block-grid lines in the spectrum (`notch_grid`; no
gain on CCPD, slightly worse on LPBlur), angular detrending of the cepstrum map, Hann instead of Tukey
windowing.

## 4. Richardson–Lucy settings

* **Boundaries.** Periodic RL produces strong boundary ringing on real crops (experiment 1: 14.2 dB
  vs. 21.4 dB for the unknown-boundary variant at 30 iterations, true PSF). Kept: masked RL on a padded
  canvas.
* **Iterations.** The best iteration count depends on noise (experiment 1). Synthetic sweeps use the
  discrepancy principle (stop when the residual reaches the estimated noise level). For real plates
  the proposal's fixed 30 iterations was kept: on LPBlur dev, 60 iterations gained only +0.01 and on
  CCPD-Blur more iterations were worse; TV regularisation gave +0.01.
* **Wiener** is a comparison only (proposal); on LPBlur dev with NSR 0.03 it was on par with RL-30.

## 5. Gate signals and thresholds

Candidate signals, all computed from the estimator's own output:

* dip depth against the whole map (`confidence`) — the proposal's signal;
* peak ratio against the best rival dip;
* **directional contrast** `angular_z` — the winning dip against all other directions at the same
  lag; isotropic structure (defocus, soft/band-limited content) is deep in every direction;
* **split-half agreement** — how many of the four image halves reproduce the estimate (a uniform blur
  is the same everywhere).

On real plates, dip depth alone ranked helpful restorations poorly (ROC AUC ≈ 0.6 on LPBlur dev),
while directional contrast and agreement were clearly better. On CCPD-Blur, where ~80% of
restorations *hurt* because the blur is mostly not uniform linear motion, only split-half agreement
separated the few helpful cases. Thresholds are chosen by `04_gate_calibration.py`: keep as many
helpful restorations as possible subject to harmful accepts in at most 5% of the images of every
development set (synthetic, LPBlur, CCPD). They are frozen in `textdeblur.gate.DEFAULT_GATE`.

## 6. Evaluation choices

* **Matched crop size.** Plates are small. Comparing a 96×256 plate with a full 512×768 Kodak frame
  confounds content with region size, so experiment 2 also cuts natural images to the same window.
* **OCR as instrument only.** RapidOCR recognition model on corner-rectified plates, six alphanumerics
  scored (province glyph dropped). On sharp CCPD plates it reads ~100% of characters, so changes in
  accuracy reflect the image, not the reader.
* **LPBlur reference text** is the OCR of the paired sharp image (the dataset has no labels); pairs whose
  sharp image does not give six characters are excluded.
* **Köhler** kernels are curved and the blur mildly non-uniform; the benchmark is used as a reference
  point for how far a line model goes on real camera shake, with reference kernels obtained by
  regularised Fourier division of the blurred image by a sharp frame. Ground truth of scenes 2 and 3
  could not be fully downloaded (archive host truncated the transfer), so PSNR uses scenes 1 and 4;
  kernel comparisons use all 48 images.
