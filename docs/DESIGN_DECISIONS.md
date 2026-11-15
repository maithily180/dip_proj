# Design decisions

What was tried, what was kept, and why. Every choice below was made on **development data**
(synthetic dev split, LPBlur dev subset, CCPD-Blur dev subset). The reported results in the report
come from test images that none of these decisions saw. `experiments/00_design_dev.py`
regenerates the development evidence for decisions 3 and 4; `experiments/04_gate_calibration.py`
regenerates decision 5.

## 1. Radon angle with a joint refinement (and a correction to phase 0)

*Phase 0* suggested that picking the angle as the direction of maximum Radon-projection variance is
fragile, because character strokes also produce high-variance projections, and that an
unrestricted joint search over the whole (direction × lag) cepstrum map would be more reliable.
The library was first built that way.

The first full test-split sweep contradicted it: the textbook two-stage rule beat the unrestricted
joint search on every class. Rather than pick the winner on test data, the comparison was redone on
the **development split** (synthetic, 3,840 trials per variant):

| angle strategy | success | mean angle error |
|---|---|---|
| unrestricted joint search | 85.5% | 5.5° |
| textbook two-stage (Radon-variance angle, then cepstral length) | 89.5% | 1.4° |
| joint search within ±8° of the Radon angle | 88.9% | 1.6° |
| **joint search within ±3° of the Radon angle** | **90.7%** | **1.3°** |

The unrestricted search fails by locking onto isolated deep dips in a wrong direction; the
variance rule integrates the whole spectrum and is the more robust angle cue. Kept: Radon-variance
angle, cepstral length, joint refinement within ±3° (`BASELINE = EstimatorConfig(angle_window=3.0)`).
This is the proposal's pipeline as written. The two alternatives remain as reference rows in
experiments 0 and 2 (`textdeblur/radon_angle.py`, `angle_window=None`).

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

Development evidence with the final code (`experiments/00_design_dev.py`; synthetic dev split,
3,840 trials per variant; LPBlur dev n = 147 and CCPD-Blur dev n = 150, OCR character accuracy after
RL-30 with each variant's estimate):

| variant (in front of the baseline estimator) | synthetic success | LPBlur OCR | CCPD-Blur OCR |
|---|---|---|---|
| blurred input (no restoration) | – | 0.234 | 0.768 |
| plain grayscale (baseline) | 90.7% | 0.280 | 0.256 |
| CLAHE | 87.8% | 0.280 | 0.234 |
| local contrast normalisation (LCN) | 91.4% | 0.289 | 0.339 |
| homomorphic filtering | 90.5% | 0.253 | 0.274 |
| **LCN + Laplacian (the proposed stage)** | **46.4%** | 0.239 | 0.584* |
| SNR-adaptive band limit | 79.7% | 0.332 | 0.171 |
| band limit 0.5 | 75.1% | 0.463 | 0.137 |
| band limit 0.3 | 55.4% | 0.637 | 0.098 |
| **multi-band search {1, 0.5, 0.3}** | **87.6%** | **0.515** | 0.214 |

\* It scores well on CCPD only because it returns very short kernels (median 8 px), so the
restoration barely changes the image; it is the worst variant at actually estimating blur.

* On synthetic data the contrast operators are neutral (LCN +0.7 points), but the proposed
  edge-enhanced combination roughly halves estimation success: the Laplacian amplifies the
  high-frequency band where noise dominates and the sinc nulls are weakest.
* On real LPBlur plates none of the contrast operators moves OCR accuracy noticeably.

The analysis of the failures pointed the opposite way. Real plate crops are small, upsampled and
JPEG-compressed: above some radius their spectrum carries no genuine image signal, only noise and
compression residue that bury the blur nulls, and that also corrupt the Radon-variance angle (the
full-band baseline returns implausibly long kernels on LPBlur, median 51 px). Restricting the band
fixes both: a limit of 0.3 Nyquist raises LPBlur OCR from 0.28 to 0.64.

A fixed limit of 0.3 Nyquist **destroys short blurs**, though: the first null of an L-pixel blur lies
at 2/L of Nyquist, so for L ≤ 9 no null falls inside the band (synthetic dev success at L = 5: 0%).
The multi-band search runs the estimator at limits {1, 0.5, 0.3} and keeps the estimate whose dip is
most *directional* (angular z-score): 87.6% synthetic success (−3.1 points against the full band) and
0.515 on LPBlur (+0.235). Selecting the band by dip depth instead of directionality picked the full
band too often. A relaxed rule favouring low bands gained at most +0.05 on LPBlur, at a further
synthetic cost, so the plain argmax was kept. An unrestricted joint search inside the multi-band
stage gave 0.553 on LPBlur but 84.0% on synthetic data; the Radon-angle version was kept for
consistency with the baseline.

**Final:** `TEXT_AWARE = BASELINE.with_(cutoff=(1.0, 0.5, 0.3))`.

CCPD-Blur is harmed by every variant (≈ 85% of images get worse): its blur is mostly not uniform
linear motion. That is the gate's job, not the estimator's.

Also tried and rejected: clipping the JPEG 8×8 block-grid lines in the spectrum (`notch_grid`; no
gain on CCPD, slightly worse on LPBlur), angular detrending of the cepstrum map, Hann instead of Tukey
windowing.

## 4. Richardson–Lucy settings

* **Boundaries.** Periodic RL produces strong boundary ringing on real crops (experiment 1: 14.2 dB
  vs. 21.4 dB for the unknown-boundary variant at 30 iterations, true PSF). Kept: masked RL on a padded
  canvas.
* **Iterations.** The best iteration count depends on noise (experiment 1). Synthetic sweeps use the
  discrepancy principle (stop when the residual reaches the estimated noise level). For real plates
  the proposal's fixed 30 iterations was kept: in a first development run (earlier estimator), 60
  iterations gained only +0.01 on LPBlur and were worse on CCPD-Blur; TV regularisation gave +0.01.
* **Wiener** is a comparison only (proposal); on the LPBlur test set with NSR 0.03 it ties RL-30.

## 5. Gate signals and thresholds

Candidate signals, all computed from the estimator's own output:

* dip depth against the whole map (`confidence`) — the proposal's signal;
* peak ratio against the best rival dip;
* **directional contrast** `angular_z` — the winning dip against all other directions at the same
  lag; isotropic structure (defocus, soft/band-limited content) is deep in every direction;
* **split-half agreement** — how many of the four image halves reproduce the estimate (a uniform blur
  is the same everywhere).

ROC AUC for "the restoration will help" (development data, final estimator):

| dev set | dip depth | directional contrast | peak ratio | split-half agreement |
|---|---|---|---|---|
| synthetic | 0.97 | 0.91 | 0.94 | 0.93 |
| LPBlur | 0.79 | 0.64 | 0.80 | 0.60 |
| CCPD-Blur (4 helpful of 150) | 0.58 | 0.47 | 0.69 | 0.45 |

On CCPD-Blur ~87% of restorations *hurt* because the blur is mostly not uniform linear motion; no
signal ranks the four helpful cases, so the useful question there is "is this a uniform line blur at
all?", which split-half agreement asks.

**Calibration rule.** The first rule tried was "keep the most helpful restorations subject to harming
at most 5% of the images of every dev set". The CCPD constraint then forces a gate that keeps only a
quarter of the helpful LPBlur restorations. A second attempt (share of the oracle gain) was unstable
because CCPD has almost no positive gain to normalise by. The final rule scores each candidate on each
dev set by its **net share** = Σ gain of accepted cases / Σ |gain| of all cases (in [−1, 1]; refusing
everything scores 0) and takes the best mean over the three sets. Result, frozen in
`textdeblur.gate.DEFAULT_GATE`:

    accept  ⇔  dip depth ≥ 4.4σ  and  ≥ 2 of 4 image halves agree

The 5%-harm operating point is kept as `STRICT_GATE` (dip depth ≥ 4.2σ, directional contrast ≥ 4σ,
≥ 3 halves) and reported alongside. Using larger overlapping windows (2/3 of the region) for the
agreement test was also tried on dev (with the earlier, unrestricted estimator); it shifted the operating point along the same trade-off curve
without improving it, so plain halves were kept.

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
