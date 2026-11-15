# Execution Plan

This plan turns the course proposal (`PROJ_PROP/main.tex`) into concrete, checkable work.
Each milestone maps to a proposal objective and ends with an exit criterion that is tested
by a script in `experiments/`, so every number in the report can be regenerated.

## Requirements traced from the proposal

| # | Proposal requirement | Where it is satisfied |
|---|---|---|
| O1 | Core pipeline: Radon angle, cepstral length, Richardson-Lucy; validated against known kernels on synthetic blur and the Köhler benchmark | `textdeblur/` core modules, `experiments/01_*`, `experiments/06_*` |
| O2 | Controlled sweep over angle x length x noise quantifying the natural-vs-text gap | `experiments/02_characterization_sweep.py` |
| O3 | Text-contrast pre-processing stage, measured on the same sweep | `textdeblur/preprocess.py`, `experiments/03_*` |
| O4 | Confidence gate; how often it avoids harm vs. withholds a good restoration | `textdeblur/gate.py`, `experiments/04_*` |
| O5 | Full pipeline on real plates (CCPD), OCR accuracy before vs. after, against the baseline | `experiments/05_*` |
| D1 | Working implementation in Python | `textdeblur/`, `demo/` |
| D2 | Quantitative evaluation report | `docs/report/` + `results/` |
| D3 | Qualitative gallery, including gate refusals | `results/figures/gallery_*` |
| D4 | Codebase with a reproducible evaluation script | `experiments/run_all.py` |
| D5 | Final presentation and report | `docs/report/`, `docs/slides/` |

Scope limits carried over from the proposal: one spatially uniform kernel per image;
Richardson-Lucy is the primary restorer and Wiener only a comparison; Radon and cepstrum are
treated as one combined estimator, not benchmarked against each other; natural-image contrast
data comes from Kodak, not the classic test images.

## Data

| Tier | Source | Role |
|---|---|---|
| Synthetic | sharp CCPD-Base plates, rendered printed text, Kodak | exact (angle, length) ground truth |
| Köhler | 48 real camera-shake images + recorded kernels | real-blur reference with known kernels |
| CCPD-Blur | 20,611 real blurred plates, plate text in file name | OCR before/after on real data |
| LPBlur | 10,408 real motion-blurred plates with paired sharp images | OCR and PSNR/SSIM on real motion blur |

Design choices are tuned on development splits (synthetic sources, LPBlur and CCPD-Blur dev
subsets); all reported numbers come from disjoint test splits. Gate thresholds are calibrated
on the development sets and then frozen.

## Milestones

1. **Foundations.** PSF generator, synthetic degradation, metrics, boundary-safe
   Richardson-Lucy and Wiener.
   *Exit:* deconvolving with the true PSF improves PSNR on every test condition.
2. **Baseline estimator.** Log-spectrum, Radon projections, cepstral profiles, joint
   (angle, length) search.
   *Exit:* on clean synthetic blur, angle within a few degrees and length within 2 px.
3. **Characterisation sweep.** image class x crop size x angle x length x noise x seed,
   logged per trial to CSV.
   *Exit:* success rates per class with the failure pockets identified.
4. **Text-contrast pre-processing.** Candidates evaluated on the identical sweep.
   *Exit:* a table stating whether, where and by how much each variant helps.
5. **Confidence gate.** Score definition, calibration, confusion matrix, harm avoided vs.
   good restorations withheld; semi-blind fallback.
   *Exit:* gating measurably reduces outputs that are worse than the input.
6. **Real data.** Köhler (kernel accuracy, restoration quality), CCPD-Blur (OCR and
   no-reference sharpness), baseline vs. text-aware pipeline.
7. **Consolidation.** Demo application, figures, galleries with failures, report, slides,
   tests, reproducibility script.

## Status (final)

All seven milestones are complete; `docs/DESIGN_DECISIONS.md` records what changed on the way
(Radon angle kept after all, band limiting instead of edge enhancement, gate calibration rule),
and the report (`docs/report/report.pdf`) gives the results per objective.

## Evaluation protocol

Every results table carries the same reference rows: blurred input (floor), restoration with
the true PSF (ceiling), baseline pipeline, plus pre-processing, plus gate, full pipeline, and
Wiener in place of Richardson-Lucy. Means are reported with spread over seeds and images.
Negative results are reported as measured.
