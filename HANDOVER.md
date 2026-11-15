# Handover: everything done from 14 to 15 November 2026

This is the detailed record of the implementation work: what was built, what was run, what was
found, what changed on the way, and what is left. It is written so that whoever takes over can
continue without asking. Shorter entry points:

* [`README.md`](README.md): setup, quick start, results at a glance
* [`docs/report/report.pdf`](docs/report/report.pdf): final report (16 pages)
* [`docs/slides/slides.pdf`](docs/slides/slides.pdf): presentation (16 slides)
* [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md): every design choice and the evidence for it
* [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md): requirements traced from the proposal

---

## 1. Where things stand

**Status: complete.** All five proposal objectives (O1–O5) and all deliverables (D1–D5) are done:

| Proposal item | Done in | State |
|---|---|---|
| O1 core pipeline, validated on known kernels and Köhler | `textdeblur/`, experiments 01 and 06 | done |
| O2 angle × length × noise sweep, natural vs. text/plates | experiment 02 | done |
| O3 text-aware pre-processing on the same sweep | experiments 00 and 03 | done (main finding is negative, see §5) |
| O4 confidence gate: harm avoided vs. good withheld | experiment 04 | done |
| O5 real plates, OCR before/after vs. baseline | experiment 05 | done |
| D1 implementation | `textdeblur/`, `demo/` | done, 66 tests pass |
| D2 quantitative report | `docs/report/` | done |
| D3 qualitative gallery incl. gate refusals | experiment 07, `results/figures/07_*` | done |
| D4 reproducible evaluation script | `experiments/run_all.py` | done |
| D5 report and slides | `docs/report/report.pdf`, `docs/slides/slides.pdf` | done (both review and rehearse) |

Nothing is running. The working tree is clean and pushed. The code needs no further work for
submission; what is left is presentation work and optional extensions (§9).

---

## 2. Timeline

| When (IST) | What |
|---|---|
| 14 Nov, 16:43 | Project committed and pushed to GitHub as-is (proposal, master plan, phase-0 sanity check). |
| 14 Nov night | Read the proposal, master plan and phase-0 code; wrote `docs/EXECUTION_PLAN.md`. Found and downloaded the datasets (§4). |
| 15 Nov, 04:51 | Core library committed: PSF model, Radon–cepstrum estimator, RL/Wiener, gate, pipeline, metrics, OCR wrapper, 60 tests. |
| 15 Nov, 05:00–09:00 | Experiment 01 (true-PSF check) run; first development runs on real plates (LPBlur, CCPD-Blur). Found that band limiting helps real plates and that CCPD-Blur is mostly not linear motion. |
| 15 Nov, 09:06 | Experiment scripts, demo app, fast Radon transform committed. |
| 15 Nov, 09:00–14:30 | Multi-band "text-aware" estimator chosen on dev data; new gate signals (directional contrast, split-half agreement); first gate calibration; experiments 04, 05, 07 written; demo tested. |
| 15 Nov, 14:30–16:00 | First test-split sweep showed the textbook Radon angle beats our unrestricted joint search. Re-checked on dev data and switched the core estimator (§5.1). Re-ran design evidence and gate calibration. |
| 15 Nov, 16:00–18:30 | Final runs of experiments 02, 03, 04 (held-out evaluation), 05, 06, 07, 01; report and slides written and compiled; README results. Last commit 18:28. |

Commits, oldest first:

```
cc2a9d7  Add DIP project: master plan, proposal, and phase 0 sanity check
7af4652  Add core library: PSF model, Radon-cepstrum estimator, RL/Wiener deconvolution, gate, pipeline
658b867  Add experiment scripts, demo app, fast Radon transform and noise-adaptive RL stopping
ce238c5  Add multi-band text-aware estimator, directional and split-half gate tests, experiments 04/05/07, demo samples
deaa34a  Calibrate the gate on development data (dip depth + split-half agreement), add strict operating point, ...
e1e75cd  Use the Radon-variance angle with a +-3 degree joint refinement as the core estimator; ...
069827b  Run experiments 00-05 on the final configuration; report sections for design, sweep, ...
82a85a5  Complete the evaluation: Koehler benchmark, galleries, final report and slides, README results
```

---

## 3. What was built

### 3.1 Library `textdeblur/` (pure NumPy / SciPy / OpenCV)

| Module | Contents |
|---|---|
| `psf.py` | `motion_psf(length, angle)`: area-sampled line kernel, exact box on the axes; `angle_diff` |
| `blur.py` | convolution, noise (with 8-bit quantisation), crops |
| `spectrum.py` | Tukey/Hann windows, mean removal, zero padding to a square grid, log-spectrum |
| `cepstrum.py` | 2-D cepstrum and polar/Radon cepstrum helpers (used in tests and analysis) |
| `estimate.py` | the estimator. `EstimatorConfig`, `BlurEstimate`, `estimate_blur`, fast `radon_transform` (rotate-and-sum with OpenCV, matches skimage), `radon_variance`, band limits (`cutoff` = number, `"auto"` or a tuple for multi-band), JPEG-grid clipping. Presets `BASELINE` and `TEXT_AWARE`. |
| `radon_angle.py` | textbook two-stage rule (Radon-variance angle, then cepstral length), kept as a reference |
| `preprocess.py` | the pre-processing operators that were evaluated (CLAHE, unsharp, high-pass, Laplacian, DoG, gradient magnitude, LCN, homomorphic, channel, `text` = LCN + Laplacian) |
| `deconv.py` | unknown-boundary (masked) Richardson–Lucy with discrepancy-principle stopping and optional TV; Wiener with edge taper; Donoho–Johnstone noise estimate |
| `gate.py` | `GateConfig`, `decide`, `consistency` (split-half agreement), ROC/AUC/calibration helpers. Presets `DEFAULT_GATE`, `STRICT_GATE`. |
| `pipeline.py` | `deblur` (blind: estimate, gate, restore), `deblur_with_psf` (semi-blind), `restore` |
| `metrics.py` | angle/length error, PSNR, SSIM, kernel similarity, sharpness, edit distance, OCR character accuracy |
| `ocr.py` | plate rectification from corner annotations and a RapidOCR reader; evaluation only, never part of the pipeline |
| `data.py` | image I/O, CCPD file-name parser, Kodak paths, deterministic rendered text pages, stable hashing for splits |

**Final configuration**

* `BASELINE = EstimatorConfig(angle_window=3.0)`: Radon-variance angle, cepstral length, joint refinement
  within ±3° of the Radon angle, full frequency band.
* `TEXT_AWARE = BASELINE.with_(cutoff=(1.0, 0.5, 0.3))`: the same search at band limits 1, 0.5 and
  0.3 of Nyquist; the estimate with the most *directional* dip (`angular_z`) wins.
* `DEFAULT_GATE = GateConfig(min_confidence=4.4, min_agreement=2)`: dip depth ≥ 4.4 robust σ and at
  least 2 of the 4 image halves reproduce the estimate.
* `STRICT_GATE = GateConfig(min_confidence=4.2, min_angular_z=4.0, min_agreement=3)`: the operating
  point that harms at most 5% of images on every development set.
* Restoration: RL, 30 iterations on real plates; discrepancy stopping (τ = 1) on synthetic data.

### 3.2 Experiments `experiments/`

| Script | Purpose | Main outputs |
|---|---|---|
| `common.py` | splits, synthetic trial generator, process pool, plotting defaults | – |
| `sweep.py` | shared synthetic sweep engine (estimates, optional restoration and agreement per variant) | – |
| `real_data.py` | LPBlur / CCPD-Blur item lists (dev/test), plate region, loading | – |
| `00_design_dev.py` | dev-split evidence behind every design choice | `00_design_dev_*.csv(.gz)` |
| `01_true_psf_check.py` | RL/Wiener with the true PSF: iterations vs. noise, boundary handling | `01_true_psf.*` |
| `02_characterization_sweep.py` | O2 sweep: natural (full frame and crop), text, plates; three angle strategies | `02_*` |
| `03_preprocess_comparison.py` | O3: 16 pre-processing / band variants on the test sweep | `03_*` |
| `04_gate_calibration.py` | O4: gate signals, calibration on dev, held-out synthetic evaluation (`--reuse` skips the dev runs) | `04_*`, `results/gate_calibration.json` |
| `05_real_plates.py` | O5: LPBlur + CCPD-Blur test sets, 7 systems, OCR, PSNR/SSIM | `05_*` |
| `06_kohler.py` | Köhler benchmark: reference kernels, line fits, benchmark PSNR (`--plot-only` redraws) | `06_*` |
| `07_gallery.py` | worked example and rule-selected real-plate galleries | `07_*` |
| `run_all.py` | runs 00–07 in order, writes logs to `results/logs`, scrubs local paths | – |

### 3.3 Other pieces

* `demo/app.py`: Streamlit app with Automatic, Semi-blind and Simulate modes, gate sliders, and
  the frequency-domain evidence (log-spectrum, Radon–cepstrum map, profile). `demo/samples/`
  holds 7 CCPD-derived and rendered samples (CCPD is MIT-licensed; LPBlur images are not shipped).
  Run with `streamlit run demo/app.py`.
* `tests/`: 66 tests (`pytest`), including a smoke test of every demo mode.
* `scripts/`: `fetch_ccpd_subset.py` (streams the 13 GB CCPD archive, keeps only what is needed),
  `prepare_kohler.py`, `make_demo_samples.py`, `scrub_paths.py`.
* `docs/report/`: LaTeX sources split by section (`report.tex` includes the rest) and the compiled
  PDF. Built locally with MiKTeX: `pdflatex report && bibtex report && pdflatex report && pdflatex report`.
* `docs/slides/`: Beamer (metropolis theme) source and PDF.

---

## 4. Data (not in the repo; everything goes under `data/`)

| Dataset | What we have | How it was obtained |
|---|---|---|
| Kodak | 24 images | `https://r0k.us/graphics/kodak/kodak/kodimNN.png` |
| CCPD2019 | all of `ccpd_blur` (20,611) and 3,280 images of `ccpd_base` (1 in 60, by file-name hash) | the Zenodo copy was too slow (~300 KB/s); the Hugging Face mirror `htkien95/ccpd2019` streams at ~11 MB/s through `scripts/fetch_ccpd_subset.py`, so the archive is never stored |
| LPBlur | 10,408 blurred/sharp pairs (350×150) | `LPBlur.zip` from the LPDGAN repository (github.com/haoyGONG/LPDGAN) |
| Köhler et al. 2012 | 48 blurred images, recorded kernels, ground-truth frames for scenes 1 and 4 | the original host is dead; files come from the Wayback Machine (`web.archive.org/web/2024id_/http://people.kyb.tuebingen.mpg.de/rolfk/BenchmarkECCV2012/...`). Scenes 2 and 3 ground-truth archives stop part-way and cannot be resumed, so Köhler PSNR uses 24 of the 48 images. |

An attempt to download CCPD on the IIIT Ada cluster failed (partition and stream errors), so all
data was fetched and processed locally. Nothing runs on Ada; a small unused `~/dip_data` folder is
left there.

**Splits.** Dev and test never overlap. Synthetic sources: Kodak 8 dev / 16 test, rendered text
8 / 16, sharp CCPD plates 8 / 24. Real data: 150 dev images each from LPBlur and CCPD-Blur; the
test sets (400 each) are drawn from the remainder. Every design choice was made on dev data.

---

## 5. What changed compared with the proposal and phase 0 (important for the presentation)

### 5.1 Radon angle kept after all (phase-0 correction)

Phase 0 suggested replacing the Radon-variance angle by an unrestricted joint search over
(direction × lag), and the library was first built that way. The first test-split sweep showed the
textbook two-stage rule doing better. To avoid choosing on test data, the comparison was redone on
the dev split: joint search 85.5%, two-stage 89.5%, Radon angle with ±3° joint refinement 90.7%
(mean angle error 5.5°, 1.4°, 1.3°). The final estimator uses the Radon angle, which is the
proposal's pipeline as written. The report states openly that a test result triggered this check.

### 5.2 The proposed pre-processing does not work; band limiting does

The proposal planned local contrast normalisation + edge enhancement before estimation. Measured:

* Synthetic test sweep: LCN + Laplacian drops success from 89.7% to 46.5%. Laplacian alone gives
  58.5%, gradient magnitude 21.8%. Contrast normalisation alone (CLAHE, LCN, homomorphic) is neutral
  to slightly harmful.
* A mild Gaussian high-pass or difference of Gaussians gains +2.5 points on synthetic data. It was
  not in the real-plate dev comparison, so it was reported but not adopted.
* Real plates (LPBlur dev): contrast operators do not move OCR accuracy. Restricting the spectrum to a
  low band raises it from 0.28 to 0.64, because plate crops are small, upsampled and JPEG-compressed:
  the upper spectrum is noise and compression residue. A fixed 0.3 limit, however, misses short
  blurs entirely, so the final stage searches three bands and keeps the most directional dip.

So the "text-aware stage" in the final system is the multi-band search, not edge enhancement. The
report and slides present this as a measured negative result plus what replaced it.

### 5.3 Gate signals and calibration rule

The proposal's single confidence score (dip depth) ranks well on synthetic data but poorly on
CCPD-Blur. Two new signals were added: directional contrast (is the dip in one direction only?) and
split-half agreement (do the four image halves give the same estimate?). The first calibration
rule ("harm at most 5% of images on every dev set") forced a gate that withheld most good LPBlur
restorations. The final rule maximises the mean *net share* (net gain let through divided by
everything at stake) over the three dev sets. The 5% point is kept as `STRICT_GATE` and reported too.

### 5.4 Extra dataset

The proposal named CCPD only. CCPD-Blur turned out to be mostly defocus and non-uniform blur, which
a straight-line model cannot fix, so LPBlur (real motion blur with sharp pairs) was added to test
O5 on blur the method is meant for. Both are reported.

---

## 6. Results (held-out test data)

**O1, true PSF (experiment 01).** RL gain depends on noise: +7.2 dB at 200 iterations without noise,
negative beyond about 20 iterations at σ = 0.04. Discrepancy stopping gives +4.8 / +2.3 / +1.5 /
+0.7 dB at σ = 0 / 0.01 / 0.02 / 0.04. Boundary handling at 30 iterations: blurred 19.4 dB, periodic
RL 14.2 dB, our unknown-boundary RL 21.4 dB.

**O2, sweep (experiment 02), success = angle within 5° and length within 2 px:**

| | natural full frame | natural crop | text crop | plate crop |
|---|---|---|---|---|
| baseline | 93.0% | 80.7% | 98.9% | 89.4% |
| textbook two-stage | 91.6% | 79.3% | 99.2% | 87.9% |
| unrestricted joint search | 90.9% | 77.0% | 95.6% | 81.6% |

Text is the easiest input; difficulty comes from region size, noise and short blur (plates: 58% at
L = 5 px, 76% at σ = 0.04).

**O3, pre-processing (experiment 03):** see §5.2. Final multi-band stage: 85.9% on synthetic data
(−3.8 points, all on short blurs).

**O4, gate (experiment 04), held-out synthetic test:** 96–100% of harmful restorations refused,
2–23% of helpful ones withheld; images made worse drop from 4–20% to at most 0.4%; mean gain on
plates rises from +1.5 to +3.0 dB.

**O5, real plates (experiment 05):**

| system | LPBlur OCR (n = 397) | LPBlur plates worse | CCPD-Blur OCR (n = 400) |
|---|---|---|---|
| blurred input | 0.176 | – | **0.759** |
| classical baseline | 0.241 | 19.1% | 0.237 |
| text-aware, no gate | **0.497** | 9.8% | 0.185 |
| text-aware + gate | 0.407 | **0.8%** | 0.684 |
| text-aware + strict gate | 0.308 | 0.3% | 0.726 |
| text-aware, Wiener | 0.497 | 9.3% | 0.215 |

LPBlur PSNR/SSIM: blurred 21.62 / 0.642, text-aware + gate 21.95 / 0.682 (best). The OCR reader
itself reads 99.8% of characters on sharp CCPD plates.

**Köhler (experiment 06), 24 images with ground truth:** blurred 24.39 dB, recorded kernel 29.24 dB,
best straight-line fit 23.42 dB, estimate 20.42 dB, estimate + gate 24.08 dB. Real hand shake is
curved, so no straight line helps here.

---

## 7. Known limitations and honest caveats (all stated in the report)

* On CCPD-Blur the gated pipeline (0.684) is still below leaving images alone (0.759). The gate
  avoids most of the damage but does not make CCPD better.
* On Köhler the gate accepts 15 of 24 images and 9 of those restorations hurt: curved shake is still
  consistent across image halves, so agreement does not detect that the line model is wrong.
* LPBlur has no text labels; the reference is the OCR of the sharp image. Pairs whose sharp image
  does not give 6 characters are excluded (3 of 400).
* Köhler ground truth only for scenes 1 and 4.
* Some development-only checks (iteration count 60 vs. 30, TV, larger agreement windows) were done
  with the earlier estimator; this is marked where cited.

---

## 8. How to run things

```bash
python -m venv .venv
. .venv/Scripts/activate              # Windows Git Bash; source .venv/bin/activate on Linux/macOS
pip install -e ".[ocr,demo,dev]"
pytest                                # 66 tests, about 1-2 minutes
streamlit run demo/app.py             # demo
python experiments/run_all.py         # everything, about 3 hours
python experiments/run_all.py 05 07   # selected experiments
```

Practical notes:

* **Run experiments one at a time.** Each uses a process pool of (CPU count − 4) workers at up to
  ~1.3 GB each. Two at once ran a 32 GB machine out of memory (experiment 2 crashed with
  `_ArrayMemoryError`, RapidOCR failed with "bad allocation"). Set `TEXTDEBLUR_WORKERS` to use fewer workers.
* Approximate run times on a 22-thread laptop: 00 ≈ 25 min, 01 ≈ 5, 02 ≈ 32, 03 ≈ 50, 04 ≈ 22,
  05 ≈ 19, 06 ≈ 25, 07 ≈ 3.
* `03_preprocess_comparison.py --plot-only` and `06_kohler.py --plot-only` redraw figures from the
  saved CSVs; `04_gate_calibration.py --reuse` reuses the dev CSVs.
* If you change `TEXT_AWARE` or the gate, rerun 04 (full), then copy its printed thresholds into
  `DEFAULT_GATE` / `STRICT_GATE` in `textdeblur/gate.py` (the script prints MATCH or DIFFERENT), then
  rerun 03, 04 `--reuse`, 05, 06, 07.
* On Windows set `PYTHONIOENCODING=utf-8` (or use `run_all.py`, which does) because OCR output and
  "Köhler" are not cp1252.
* Large per-trial tables are stored as `.csv.gz`; pandas reads them directly.

---

## 9. What is left / suggested next steps

For the submission:

1. Read the report and slides end to end and adjust wording or emphasis to taste; rehearse the talk.
   The talk outline is the slide order: problem → method → O1 → O2 → O3 → O4 → O5 → galleries →
   Köhler → takeaways → demo.
2. Prepare a live demo: `streamlit run demo/app.py`; "Simulate" mode on `sharp_plate_1` shows the
   true-vs-estimated PSF clearly, and "Automatic" on `ccpd_blur_real_*` shows gate refusals.

Optional extensions, in order of expected value:

1. Test the mild Gaussian high-pass / DoG pre-filter on the real-plate dev sets (it helped synthetic
   data by +2.5 points and was not adopted only because it had not been checked on real plates).
2. A gate test of the line hypothesis itself, e.g. comparing the observed cepstral pattern with the
   one predicted by the fitted line kernel, to catch curved shake (Köhler) and defocus.
3. A two-segment or curved kernel model using the same spectral evidence.
4. Retry the Köhler scene 2/3 ground-truth downloads to report PSNR on all 48 images.
