# Seeing Through the Shake
## Text-Aware PSF Estimation for Blind Motion Deblurring of License Plates and Low-Texture Text

**Course:** Digital Image Processing
**Team:** Inesh Roy (2024112015), Maithily Gauravkumar Bhala (2024117011)
**Document type:** Master project plan / technical blueprint
**Status:** Living document — update as phases complete

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Why This Project (Positioning & Pitch)](#2-why-this-project-positioning--pitch)
3. [Theoretical Background](#3-theoretical-background)
4. [Literature Review & Reference Map](#4-literature-review--reference-map)
5. [Datasets](#5-datasets)
6. [Code, Libraries & Repositories](#6-code-libraries--repositories)
7. [System Architecture](#7-system-architecture)
8. [Phase Plan](#8-phase-plan)
9. [Evaluation Protocol](#9-evaluation-protocol)
10. [Risk Register & Mitigations](#10-risk-register--mitigations)
11. [Demo & Presentation Strategy](#11-demo--presentation-strategy)
12. [Repository Structure](#12-repository-structure)
13. [Glossary](#13-glossary)
14. [Full Bibliography](#14-full-bibliography)

---

## 1. Executive Summary

### 1.1 One-paragraph version

Motion blur destroys exactly the information that makes text legible. Classical blind
deblurring can recover a sharp image without any training data by estimating the blur
kernel (PSF) directly from the image's frequency-domain signature — a Radon transform on
the Fourier spectrum gives the blur **angle**, and a cepstral analysis gives the blur
**length**. This technique is well established, but it was developed and is almost always
evaluated on natural, texture-rich photographs. License plates and printed text are a
structurally different, weaker-signal input class. This project builds a **text-aware**
version of that classical pipeline: we quantify where and why the standard approach
degrades on low-texture text, add a text-contrast pre-processing stage designed to
strengthen the specific cues the estimators depend on, and add a confidence gate that
refuses to deconvolve when the PSF estimate is unreliable (because a wrong PSF actively
damages the image rather than merely failing to help).

### 1.2 The one-sentence novelty claim

> Every component we use is an established, published technique; the contribution is
> assembling and evaluating them around a specific, underexamined failure regime —
> low-texture text — and testing whether targeted pre-processing plus confidence gating
> measurably closes the natural-image-vs-text performance gap.

### 1.3 Why it satisfies the DIP-not-CV constraint

| Stage | What it is | DIP or CV? |
|---|---|---|
| Fourier spectrum analysis | Transform-domain analysis | **DIP** (core) |
| Radon transform for angle | Integral transform | **DIP** (core) |
| Cepstral analysis for length | Homomorphic / log-spectral analysis | **DIP** (core) |
| Text-contrast pre-processing | Local contrast normalization, edge enhancement | **DIP** (core) |
| Richardson–Lucy deconvolution | Iterative image restoration | **DIP** (core) |
| Confidence gating | Signal-to-background thresholding on cepstral peak | **DIP** (core) |
| OCR | *Evaluation metric only* | CV — but **not in the pipeline** |

**Critical framing point for the presentation:** OCR is never part of our algorithm. It is
purely a *measuring instrument* used to answer "did our restoration make the text more
readable?" We could delete OCR entirely and the pipeline would still function. Say this
explicitly if challenged — it's the single most likely objection.

---

## 2. Why This Project (Positioning & Pitch)

### 2.1 The problem in plain language

You photograph a moving car. The plate is a smear. Can you recover the number using only
classical signal processing — no neural network, no training data, no GPU?

### 2.2 Why the naive answer ("just use a CNN") is the wrong answer here

- Requires large paired blurred/sharp datasets.
- Black box — you cannot explain *why* it worked.
- Wrong fit for a DIP course.
- **Crucially:** the LPBlur paper (2024) demonstrated that deep models trained on
  *synthetic* blur transfer poorly to *real* license-plate blur. This is a useful citation —
  it shows even the deep-learning route has an unsolved data problem, which strengthens
  the case for an interpretable model-based approach.

### 2.3 The specific gap we target

Classical Radon+cepstrum PSF estimation assumes the image gives the estimator a strong
directional signal in the frequency domain. Natural photos do. Text does not:

| Property | Natural photo | License plate / text |
|---|---|---|
| Texture density | High, broadband | Sparse, few strokes |
| Dominant directions | Many | Few, often axis-aligned |
| Contrast | Varied | High local, low global |
| Region size | Full frame | Small crop |
| Frequency content | Rich | Concentrated, near-periodic |

The last row is the dangerous one: **text has quasi-periodic structure** (evenly spaced
characters), and periodic structure produces cepstral peaks that have nothing to do with
blur. This is the theoretical reason we expect the standard pipeline to struggle here, and
it's the mechanism our pre-processing stage must work around.

### 2.4 How to pitch this in 30 seconds to the professor

> "Everyone reaches for a neural network for deblurring. We're showing you can do it with
> pure transform-domain DIP — Fourier, Radon, cepstrum, Richardson–Lucy — no training
> data at all. And we're targeting the case that classical methods are documented as being
> *worst* at: small, low-texture text like license plates. We add a pre-processing stage to
> fix the weak-signal problem and a confidence gate so the system knows when to refuse
> rather than make the image worse."

---

## 3. Theoretical Background

*Everything in this section should be understood well enough to whiteboard it. The
professor will ask.*

### 3.1 The blur model

Motion blur is modeled as convolution with a point spread function plus noise:

```
b(x, y) = l(x, y) * k(x, y) + n(x, y)
```

Where:
- `b` = observed blurred image
- `l` = latent (sharp) image we want
- `k` = PSF / blur kernel
- `n` = additive noise
- `*` = convolution

**Blind deblurring** = we know only `b`, must estimate both `k` and `l`.
**Non-blind deblurring** = we know `b` and `k`, must recover `l`. (Much easier.)

Our strategy: **estimate `k` first (blind step), then solve for `l` (non-blind step).**
This is the classic two-stage decomposition and is what makes the problem tractable.

### 3.2 The linear uniform motion assumption

We assume the blur is:
- **Linear** — motion along a straight line
- **Uniform** — constant velocity, same across the whole image (spatially invariant)

Under this assumption the PSF is fully described by **two scalars**:
- `θ` — blur **angle** (direction of motion)
- `L` — blur **length** (how many pixels of travel)

This reduces an infinite-dimensional estimation problem to a 2-parameter search. **This
assumption is the entire reason the project is tractable, and it is also the project's
main limitation.** Real blur has rotation, acceleration, out-of-plane motion. Say this out
loud in the report — it's standard practice in the literature, not a flaw we're hiding.

### 3.3 Why the angle shows up in the Fourier spectrum

A linear motion PSF of length `L` at angle `θ` is a 1-D box (line) function in 2-D space.
Its Fourier transform is a **sinc function** oriented perpendicular to the motion direction.

Sinc has periodic zeros. Those zeros appear in the log-magnitude spectrum of the blurred
image as **dark parallel stripes**, oriented **perpendicular to the blur direction**.

So: *find the stripe orientation in the spectrum → you have the blur angle.*

**Watch out:** the stripes are perpendicular to motion. Off-by-90° errors are the single
most common bug in implementations of this. Verify your convention on a synthetic image
with known angle before trusting anything.

### 3.4 Why the Radon transform finds those stripes

The Radon transform projects a 2-D image onto a 1-D line at every angle — it integrates
along parallel lines. When the projection direction aligns with the stripes:

- Projecting *along* the stripes → they reinforce → distinctive projection profile
- Projecting *across* the stripes → they average out → flat profile

By sweeping θ from 0° to 180° and scoring each projection, you find the stripe direction.

**Scoring rule matters enormously.** The textbook rule ("pick the angle minimizing
projection variance") is fragile because the image's *own* dominant edge directions
compete with the blur stripes. Consider scoring instead by cepstral-echo strength (see
§3.6) — for each candidate angle, compute how strong a cepstral dip appears along that
direction, and pick the angle with the strongest dip. This jointly estimates angle and
length and is more robust in practice.

### 3.5 Why the length shows up in the cepstrum

The **cepstrum** is:

```
C(x, y) = IFFT( log( |FFT(b(x,y))| ) )
```

Reasoning:
1. Convolution in space = **multiplication** in frequency: `B = L · K`
2. Taking log converts multiplication to **addition**: `log|B| = log|L| + log|K|`
3. Inverse-transforming this gives a domain where blur contributes an **additive,
   localized** signature rather than being entangled multiplicatively with image content.

For a linear motion blur of length `L`, the PSF's log-spectrum is periodic (from the sinc
zeros), and that periodicity produces a **negative peak (dip) in the cepstrum at distance
`L` from the origin**, along the blur direction.

So: *sample the cepstrum along the estimated angle, find the strongest negative dip,
its distance from origin = blur length.*

This is the same homomorphic-filtering idea from the textbook (separating multiplicatively
combined signals via the log domain) — worth pointing out in the report, since it connects
directly to standard DIP course material.

### 3.6 Joint estimation (recommended approach)

Rather than estimating angle first, then length:

```
best_score = -inf
for θ in 0..180 (1° steps):
    profile = cepstrum sampled along direction θ
    (dip_depth, dip_position) = strongest negative dip in profile
    if dip_depth > best_score:
        best_score = dip_depth
        best_angle = θ
        best_length = dip_position
```

This couples the two estimates and uses the cepstral evidence to disambiguate the angle,
which is more robust than scoring the Radon projection alone. It also gives you
`best_score` for free — **this is your confidence signal for the gating stage.**

### 3.7 Richardson–Lucy deconvolution

Given `b` and estimated `k`, recover `l`. RL is an iterative maximum-likelihood method
assuming Poisson noise:

```
l_{n+1} = l_n · [ (b / (l_n * k)) * k̂ ]
```

where `k̂` is `k` flipped. Properties:
- Non-negative output (physically meaningful)
- Preserves total intensity
- **More iterations = sharper but more ringing/noise amplification.** Iteration count is
  a tuning knob you must justify.

**Alternative to implement for comparison:** Wiener deconvolution — a single-shot
frequency-domain inverse filter regularized by an estimated noise-to-signal ratio. Faster,
more forgiving, but generally less sharp. Having both gives you a free, legitimate ablation
for the report.

### 3.8 The boundary artifact problem

FFT-based deconvolution treats the image as periodic. Real images aren't — the left edge
doesn't continue into the right edge. This wraparound produces **dark bands at image
borders** that can dominate pixel-wise error metrics even when the image interior is
genuinely sharper.

**Fixes (implement one, mention it in the report):**
- Edge-taper / apodization: window the image edges before deconvolution
- Symmetric/reflect padding before, crop after
- Crop borders before computing PSNR/SSIM

**Do not skip this.** It can make a working pipeline look broken in your metrics table.

### 3.9 Why a wrong PSF is worse than no PSF

Deconvolution is an inverse operation. Feeding it the wrong kernel doesn't produce a
slightly-off sharp image — it *amplifies the wrong frequencies*, creating structured
artifacts (ringing, ghosting, false edges) that can drive image quality below the blurred
input. This is the entire justification for the confidence gate: **the system must be able
to say "I don't know" rather than always acting.**

---

## 4. Literature Review & Reference Map

### 4.1 How to read this section

Papers are grouped by what you need them *for*. For each: what it gives you, and what to
actually do with it.

### 4.2 Foundational blind deblurring

**Fergus et al., "Removing Camera Shake from a Single Photograph," SIGGRAPH 2006**
- The landmark paper that made single-image blind deblurring credible.
- **Use for:** Introduction/related-work framing. Establishes the problem's importance.
- **Don't:** try to implement it — it's a variational Bayesian method, far beyond scope.

**Levin et al., "Understanding and Evaluating Blind Deconvolution Algorithms," CVPR 2009**
- Critical analysis of *how* blind deblurring methods should be evaluated. Points out that
  many methods' apparent success came from evaluation artifacts.
- **Use for:** Designing your evaluation protocol honestly. Cite when justifying why you
  evaluate against known ground-truth kernels rather than just eyeballing outputs.

### 4.3 Radon-based angle estimation (core technique)

**Wang et al., "A New Method and Implementation of Blind Restoration Algorithm for Moving
Fuzzy License Plate Image Based on Frequency-Domain Characteristics" (2021)**
- **The closest paper to our project.** Uses Radon transform to detect parallel line
  features in the spectrum of a blurred license plate to estimate blur angle, then uses
  cepstrum data to measure a peak and estimate blur length, then restores via Wiener
  filtering.
- **Use for:** Your primary methodological reference. This is essentially the baseline
  you're reimplementing and then extending.
- **Note:** It uses Wiener filtering as the restoration step; we plan Richardson–Lucy as
  primary with Wiener as comparison. That's a legitimate, small difference to note.

**Zhou et al., "Improved Estimation of Motion Blur Parameters for Restoration from a
Single Image," PLOS ONE 2020**
- Introduces a modification to the Radon transform for blur-angle estimation using a
  "difference value vs angle" curve, and uses an autocorrelation matrix to estimate blur
  parameters by measuring distance between conjugated-correlated troughs. Reports
  improved accuracy, robustness, and time efficiency versus existing algorithms.
- **Use for:** If your basic Radon scoring is too fragile, this is the first place to look
  for a better-behaved scoring rule. Also a good citation for "the scoring rule choice is a
  known, actively researched issue," which pre-empts criticism.

**Zhou et al. (Optik 2021), "A self-adaptive learning method for motion blur kernel
estimation"**
- Describes a self-adaptive Radon approach with multi-stage angle estimation.
- **Use for:** Evidence that multi-pass/adaptive angle estimation is an established idea.

### 4.4 Cepstral length estimation (core technique)

**Deshpande & Patnaik, "A Novel Modified Cepstral Based Technique for Blind Estimation of
Motion Blur," Optik 2013**
- Uses bit-plane slicing on the modified cepstrum; notes that the fourth bit plane of the
  modified cepstrum carries an important cue for estimating blur direction, requiring no
  further post-processing. Evaluated on both real-blurred photographs and synthetically
  blurred standard test images (Berkeley segmentation dataset, USC-SIPI textures).
- **Use for:** A concrete, implementable trick if your raw cepstral peak detection is too
  noisy. Bit-plane slicing is textbook DIP — a great fit for the course.

**"PSF estimation and image restoration for noiseless motion blurred images"**
- Notes that for the linear motion blur case it is sufficient to estimate motion length and
  direction, with parameters estimated via modified Radon transform and power cepstrum
  analysis. Compares Wiener vs Lucy-Richardson restoration across image layouts.
- **Use for:** Justifying the 2-parameter reduction; also a source for the Wiener-vs-RL
  comparison framing.

**"Single image motion deblurring: An accurate PSF estimation and ringing reduction"**
- Combines dual Fourier spectrum with bit plane slicing and Radon transform for accurate
  blur length/angle estimation, and proposes a post-processing method within the
  Richardson–Lucy framework to reduce ringing artifacts at boundaries and near strong
  edges.
- **Use for:** Directly relevant to §3.8 (boundary artifacts). If ringing is hurting you,
  this paper is about exactly that.

**"Blind Image Deconvolution Motion Blur Estimation"**
- Notes that combining angle estimates from the Radon transform with length estimates from
  the two-dimensional cepstral method allows refined accuracy, particularly for smaller
  blur lengths.
- **Use for:** Direct citation supporting the Radon+cepstrum combination as the standard
  pairing, and flagging small-blur-length as a known weak spot.

### 4.5 Deconvolution

**Richardson (1972), JOSA** and **Lucy (1974), Astronomical Journal**
- The two independent original derivations of the RL algorithm.
- **Use for:** Citing the deconvolution step properly. You don't need to read them deeply —
  `skimage`'s implementation is standard.

### 4.6 License-plate-specific work

**"A Dataset and Model for Realistic License Plate Deblurring" (LPBlur, arXiv 2404.13677,
2024)**
- Introduces the first large-scale license-plate deblurring dataset with *real* blur, and
  demonstrates that models trained on synthetically blurred data fail to eliminate
  real-world license-plate blur effectively — showing a significant disparity between
  synthetic and real-world blur. Also notes that blur from vehicular motion substantially
  exceeds that from camera shake.
- **Use for:** Two things. (1) A dataset source. (2) **Critical framing ammunition** — the
  synthetic-vs-real gap is documented, so when your real-image results are worse than your
  synthetic results, you have a published citation saying that's expected, not a failure
  of your implementation.

**CCPD / Xu et al., ECCV 2018 — "Towards End-to-End License Plate Detection and
Recognition: A Large Dataset and Baseline"**
- The dataset paper. See §5.

### 4.7 Benchmark

**Köhler et al., ECCV 2012 — "Recording and Playback of Camera Shake"**
- Used a Stewart platform with six degrees of freedom to replay recorded camera motion
  trajectories, capturing sharp images and long-exposure blurry images. Comprises 4 clear
  images and 48 blurry images across 12 motion trajectories.
- **Use for:** Ground-truth-kernel validation. Small enough to work with easily, real
  enough to be credible.

### 4.8 Context: why not deep learning

**Nah et al., CVPR 2017 (GoPro dataset)** — large-scale synthetic dataset generated by
averaging frames from high-frame-rate video.
**Kupyn et al., DeblurGAN / DeblurGAN-v2**
**Rim et al., ECCV 2020 (RealBlur)** — argues that GoPro-style synthetic data doesn't
cover real-world blur well.
- **Use for:** One paragraph in related work establishing what the deep-learning route
  looks like and why we're deliberately not taking it. Do not go deeper — this is a DIP
  project and dwelling on CNNs works against you.

---

## 5. Datasets

### 5.1 Synthetic (built by us) — PRIMARY DEVELOPMENT TOOL

**What:** Take sharp images, convolve with a known line kernel at known (θ, L), optionally
add Gaussian noise.

**Why it's essential:** It's the only way to get **exact ground truth for the PSF itself**.
You can measure angle error in degrees and length error in pixels. Without this you're
guessing.

**Sources of sharp base images:**
- Sharp plate crops from CCPD-Base
- Any standard test images (for natural-photo comparison baseline)
- Photos of text you take yourself

**Implementation notes:**
- Use sub-pixel line kernel generation (anti-aliased), not a crude binary line — a jagged
  kernel introduces its own artifacts.
- Normalize kernel to sum to 1.
- Sweep systematically: multiple angles × multiple lengths × multiple noise levels ×
  multiple random seeds. One lucky configuration proves nothing.

### 5.2 Köhler dataset — GROUND-TRUTH KERNEL VALIDATION

- 4 sharp images, 12 real recorded camera-shake trajectories, 48 blurred images.
- Real camera motion, controlled lab conditions, **known kernels**.
- URL: `https://webdav.tuebingen.mpg.de/pixel/benchmark4camerashake`
- **Caveat:** Köhler kernels are non-linear camera-shake trajectories, not pure linear
  motion. Your 2-parameter model will only approximate them. Expect degraded but non-zero
  performance — and *say so in advance* rather than being surprised.

### 5.3 CCPD (Chinese City Parking Dataset) — REAL PLATES

- Repo: `https://github.com/detectRecog/CCPD`
- Archived mirror: Zenodo record 15647076 (CCPD2019 + CCPD2020 subsets)
- ~280k+ vehicle images captured in uncontrolled conditions — different weather,
  illumination, rotation, and vagueness. Resolution 720×1160.
- **Subsets relevant to us:**
  - `ccpd_blur` — **the blur test subset. This is your primary real-world test set.**
  - `ccpd_base` — sharp plates, use as source material for synthetic blur
  - `ccpd_challenge`, `ccpd_db`, `ccpd_tilt` — harder conditions if you want stress tests
- Annotations include the license plate number itself (encoded in the filename), plus
  bounding box — so **OCR ground truth is free**, no manual labeling.
- MIT licensed, open source.
- **Important:** No sharp/blurred *pairs* — ccpd_blur images have no sharp counterpart. So
  you can't compute PSNR against ground truth here; you must use OCR accuracy and
  no-reference sharpness metrics instead. Plan for this.

### 5.4 LPBlur — REAL PAIRED PLATE BLUR (check availability)

- From arXiv 2404.13677 (2024). Paired real blurred/sharp license plates.
- **If publicly downloadable, this is the best dataset for the project** — it's the only
  one giving you real blur *and* sharp ground truth for the same plate, enabling real
  PSNR/SSIM on real data.
- **Action item:** verify availability early. If it's available, it changes your evaluation
  plan for the better. If not, fall back to CCPD-Blur + OCR metrics.

### 5.5 Self-captured — DEMO MATERIAL

Take your own motion-blurred photos of text/plates/signs with a phone (deliberately move
during a long exposure). Uncontrolled, no ground truth, but:
- Perfect for the live demo
- Demonstrates the method works outside curated datasets
- Free and immediate

### 5.6 Dataset summary table

| Dataset | Blur type | Sharp GT? | Kernel GT? | Use for |
|---|---|---|---|---|
| Synthetic (ours) | Linear, controlled | Yes | **Yes** | Development, sweeps, main quantitative results |
| Köhler | Real camera shake | Yes | Yes (non-linear) | Kernel-estimation validation |
| CCPD-Blur | Real, uncontrolled | No | No | Real-world evaluation via OCR |
| CCPD-Base | Sharp | N/A | N/A | Source for realistic synthetic blur |
| LPBlur | Real | Yes | No | Best real evaluation *if available* |
| Self-captured | Real | No | No | Live demo |

---

## 6. Code, Libraries & Repositories

### 6.1 Core stack

```
Python 3.10+
numpy            # array math
scipy            # FFT, signal processing, optimization
scikit-image     # radon, richardson_lucy, wiener, metrics, test images
opencv-python    # image I/O, color spaces, CLAHE, morphology
matplotlib       # all plotting and figure generation
```

Optional:
```
pytesseract      # OCR for evaluation (requires Tesseract binary)
tqdm             # progress bars for long sweeps
pandas           # results tables
seaborn          # heatmaps
```

### 6.2 The key library functions (these do the hard math for you)

| Function | What it does | Notes |
|---|---|---|
| `skimage.transform.radon(image, theta=...)` | Radon transform | Check `circle` parameter behavior |
| `skimage.restoration.richardson_lucy(image, psf, num_iter=...)` | RL deconvolution | Tune `num_iter`; more = sharper + more ringing |
| `skimage.restoration.wiener(image, psf, balance)` | Wiener deconvolution | Comparison baseline |
| `scipy.fft.fft2 / ifft2 / fftshift` | Frequency transforms | `fftshift` for centering spectra |
| `scipy.signal.convolve2d` | Applying synthetic blur | Use `mode='same'`, careful with boundary |
| `skimage.metrics.peak_signal_noise_ratio` | PSNR | Crop borders first (§3.8) |
| `skimage.metrics.structural_similarity` | SSIM | Better perceptual proxy than PSNR |
| `cv2.createCLAHE()` | Contrast-limited adaptive histogram eq. | Candidate for text pre-processing |
| `skimage.filters.threshold_sauvola` | Local adaptive thresholding | Candidate for text pre-processing |

**The significance of this table:** the two most mathematically intimidating steps (Radon,
Richardson–Lucy) are single function calls. You are not implementing numerical
optimization from scratch. Your engineering effort goes into *what you feed them* and *how
you score the results* — which is exactly where your contribution lives.

### 6.3 Reference repositories to study

**`ankitVP77/Blind-Motion-Deblurring-for-Legible-License-Plates-using-Deep-Learning`**
- Estimates PSF length/angle then applies Wiener deconvolution for license plates. Includes
  a `create_blurred.py` script that randomly blurs images with a random blur length and
  angle and generates a JSON of ground-truth labels — note it treats angles above 180° as
  cyclic since this doesn't affect the PSF.
- **Use for:** The synthetic blur generation script is a directly useful template. Their
  overall pipeline structure is a good reference.
- **Note:** They use a trained CNN for angle/length estimation. We use Radon+cepstrum
  instead — **this difference is worth stating in your report** as it distinguishes your
  approach from the most visible existing plate-deblurring code.

**`adamalavi/Blind-motion-deblurring-for-license-plates`**
- A fork/variant of the above with the same structure.

**GitHub topic pages** — browse for more classical implementations:
- `https://github.com/topics/psf-estimation`
- `https://github.com/topics/blind-deconvolution`
- `https://github.com/topics/motion-deblurring`
- One repo described on the psf-estimation topic page does "adaptive blind image deblurring
  with robust PSF estimation, artifact-aware restoration, native-resolution benchmarking"
  using FFT/dark-channel/blind-deblurring — worth inspecting.

**Awesome-list:** There's a curated list of deblurring papers and datasets referenced on
the motion-deblurring topic page — good for finding anything this document missed.

### 6.4 Repository hygiene

- Git from day one. Commit working states before experiments.
- Fix random seeds for all sweeps — results must be reproducible.
- Save every experiment's config alongside its results (JSON/CSV), not just the numbers.
- One script = one job. Don't build a monolith.

---

## 7. System Architecture

### 7.1 Pipeline diagram (textual)

```
                    ┌─────────────────────┐
                    │  Input blurred image │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Region selection    │  (crop to plate/text region;
                    │  (manual or given)   │   manual is fine — keeps it DIP)
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │  ★ TEXT-CONTRAST PRE-PROCESSING │  ← OUR CONTRIBUTION
              │  - local contrast normalization │
              │  - stroke-edge enhancement      │
              │  - (evaluated vs. plain gray)   │
              └────────────────┬────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  FFT → log-magnitude │
                    │       spectrum       │
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │  JOINT (θ, L) SEARCH             │
              │  for each θ in 0..180:           │
              │    cepstrum profile along θ      │
              │    find strongest negative dip   │
              │  → best (θ, L), dip_strength     │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │  ★ CONFIDENCE GATE               │  ← OUR CONTRIBUTION
              │  dip_strength vs. noise floor    │
              │  ├─ LOW  → skip, return original │
              │  └─ HIGH → proceed               │
              └────────────────┬────────────────┘
                               │ (high confidence)
                    ┌──────────▼──────────┐
                    │  Build PSF from θ,L  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Edge taper / pad    │  (boundary artifact fix)
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │  Richardson–Lucy deconvolution   │
              │  (Wiener as comparison branch)   │
              └────────────────┬────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Crop / post-process │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Restored image +    │
                    │  confidence report   │
                    └─────────────────────┘
```

### 7.2 The two contribution points (marked ★)

Everything unmarked is a faithful reimplementation of established technique. The two
starred blocks are where your project adds something. Keep this diagram in your slides —
it makes the contribution instantly legible to anyone.

### 7.3 Text-contrast pre-processing — candidate techniques

You will **evaluate several and pick empirically**. All are standard DIP:

| Technique | Rationale | Risk |
|---|---|---|
| **CLAHE** (contrast-limited adaptive histogram equalization) | Boosts local stroke contrast without blowing out globally | May amplify noise |
| **Unsharp masking / high-pass emphasis** | Directly strengthens edge energy that feeds the spectrum | Can create false edges |
| **Gradient magnitude image** | Feed Radon/cepstrum the edge map rather than intensities | Loses some blur signature |
| **Local contrast normalization** | Removes illumination gradient, equalizes stroke visibility | Parameter sensitive |
| **Homomorphic filtering** | Separates illumination from reflectance; plates often have glare | Extra parameters |
| **Color-channel selection** | Plate text often has better contrast in one channel than grayscale | Dataset dependent |

**Method:** Run the identical evaluation sweep with each candidate as pre-processing,
compare angle/length error against the plain-grayscale baseline. Report the comparison
table — *this table is a core deliverable*, and it's a legitimate result whichever way it
comes out.

### 7.4 Confidence gate — design

**Signal:** `dip_strength` from the joint search (§3.6).

**Approach:** Normalize dip depth relative to the local variation of the cepstral profile
(a signal-to-background ratio), giving a scale-invariant confidence score. Threshold it.

**Calibration:** Using synthetic data (where you know the true kernel), plot
`confidence_score` against `angle_error`. Choose a threshold that captures most of the
high-error cases. Report the resulting confusion matrix:

|  | Actually good PSF | Actually bad PSF |
|---|---|---|
| **Gate says proceed** | ✅ Correct restore | ❌ Harmful restore (minimize!) |
| **Gate says skip** | ⚠️ Missed opportunity | ✅ Correctly avoided damage |

**This confusion matrix is a strong, unusual result to show a professor** — most student
deblurring projects have no notion of the system knowing when it's wrong.

### 7.5 Fallback: semi-blind mode

If automatic estimation proves unreliable on real plates, add a mode where the user marks
the blur streak direction/length on the image (e.g., drag a line over a visibly smeared
point light or character edge). Then run non-blind deconvolution with that kernel.

**This is not cheating** — semi-blind deconvolution is a legitimate, published category.
And it **guarantees a working demo** regardless of how automatic estimation performs on a
given image. Build it. It's cheap and it de-risks your presentation entirely.

---

## 8. Phase Plan

*Adjust week counts to your actual deadline. The ordering matters more than the durations.*

### Phase 1 — Foundations & synthetic harness
**Goal:** Be able to blur an image by a known amount and measure how well you recover it.

Tasks:
- [ ] Set up repo, environment, dependencies
- [ ] Implement sub-pixel line-kernel PSF generator `(θ, L) → kernel`
- [ ] Implement synthetic blur: `sharp + (θ, L, noise) → blurred`
- [ ] Implement metric functions: angle error, length error, PSNR, SSIM (with border crop)
- [ ] Verify: blur an image, deconvolve with the **true** kernel, confirm PSNR improves

**Exit criterion:** Given the true PSF, Richardson–Lucy demonstrably improves image
quality. If this doesn't work, nothing downstream can. **Do not proceed until it does.**

**Common failure:** Boundary artifacts making PSNR drop even with the true kernel. Fix
per §3.8 before concluding anything is broken.

### Phase 2 — Baseline PSF estimation
**Goal:** Recover (θ, L) automatically from a blurred image.

Tasks:
- [ ] FFT log-magnitude spectrum computation + visualization
- [ ] Radon transform sweep
- [ ] Cepstrum computation + profile sampling along a given angle
- [ ] Dip detection in the cepstral profile
- [ ] Implement **both** scoring approaches:
  - Radon projection variance (textbook)
  - Joint cepstral-dip search (§3.6)
- [ ] Verify angle convention (perpendicular vs parallel — see §3.3 warning)

**Exit criterion:** On clean synthetic blur at multiple angles, recover θ within a few
degrees and L within a couple of pixels.

**Deliverable:** A comparison of the two scoring rules — this is a genuine, reportable
finding regardless of outcome.

### Phase 3 — Systematic characterization sweep
**Goal:** Quantify exactly where the baseline works and where it fails. **This phase
produces the data that justifies your entire contribution.**

Tasks:
- [ ] Build sweep harness: `image_type × angle × length × noise × seed`
- [ ] Include at minimum: natural photos AND text/plate-like images
- [ ] Suggested grid: ≥4 image types × ≥6 angles × ≥3 lengths × ≥4 noise levels × ≥5 seeds
- [ ] Log every run (config + result) to CSV
- [ ] Generate: success-rate tables, error heatmaps (angle × noise), per-image-type summary

**Exit criterion:** You can state, with numbers, "the baseline succeeds X% of the time on
natural images and Y% on text images, with failures concentrated at [specific conditions]."

**Why this matters most:** This is your motivation section, your novelty justification, and
your evaluation baseline all at once. Do not rush it.

### Phase 4 — Text-contrast pre-processing (CONTRIBUTION 1)
**Goal:** Close the gap identified in Phase 3.

Tasks:
- [ ] Implement each candidate from §7.3
- [ ] Re-run the **exact Phase 3 sweep** with each candidate inserted
- [ ] Build comparison table: baseline vs. each pre-processing variant
- [ ] Target the specific failure pockets Phase 3 identified, not just aggregate accuracy
- [ ] Analyze *why* the winner works (or why none do) — the explanation matters more than
      the number

**Exit criterion:** A clear, quantitative statement about whether text-aware
pre-processing helps, where, and by how much.

**Note:** "It didn't help much" is a legitimate scientific result *if* rigorously measured
and explained. Don't fake a win. A well-analyzed negative result with a good explanation
presents better than a suspicious positive one.

### Phase 5 — Confidence gating (CONTRIBUTION 2)
**Goal:** Make the system refuse rather than damage.

Tasks:
- [ ] Define and implement the confidence score
- [ ] Calibrate threshold on synthetic data (confidence vs. actual error scatter plot)
- [ ] Build the confusion matrix (§7.4)
- [ ] Measure: how much harmful-restoration does gating prevent? At what cost in missed
      good restorations?
- [ ] Implement the semi-blind fallback mode (§7.5)

**Exit criterion:** Gating demonstrably reduces cases where output is worse than input.

### Phase 6 — Real-world evaluation
**Goal:** Test on data you didn't construct.

Tasks:
- [ ] Köhler dataset run (real camera shake, known kernels)
- [ ] CCPD-Blur run (real plates) — evaluate via OCR accuracy + no-reference sharpness
- [ ] LPBlur run if available (real paired data → real PSNR/SSIM)
- [ ] Self-captured photos
- [ ] Compare: baseline pipeline vs. full text-aware pipeline on all of the above

**Expect degradation vs. synthetic results.** Real blur isn't linear or uniform. The LPBlur
paper documents this synthetic-vs-real gap explicitly — cite it, don't apologize for it.

### Phase 7 — Consolidation, demo & writeup
Tasks:
- [ ] Build the interactive/live demo (§11)
- [ ] Generate final figure set (before/after galleries, heatmaps, comparison tables)
- [ ] Curate failure cases — **include them**, they build credibility
- [ ] Write report
- [ ] Build slide deck
- [ ] Rehearse, including anticipated Q&A (§11.4)

### 8.1 Phase dependency graph

```
Phase 1 (harness)
   │
   └──► Phase 2 (baseline estimation)
           │
           └──► Phase 3 (characterization sweep)  ◄── THE CRITICAL PHASE
                   │
                   ├──► Phase 4 (pre-processing)   ─┐
                   │                                ├──► Phase 6 (real data)
                   └──► Phase 5 (confidence gate)  ─┘        │
                                                              └──► Phase 7 (demo/report)
```

Phases 4 and 5 are independent of each other and can be worked in parallel by two people.

### 8.2 Suggested work split (two people)

| Person | Primary ownership |
|---|---|
| **A** | Phases 1–2 (harness + estimation core), then Phase 5 (confidence gating) |
| **B** | Phase 3 (sweep infrastructure + analysis), then Phase 4 (pre-processing) |
| **Both** | Phase 6 (real data), Phase 7 (demo, report, slides) |

Rationale: A owns the algorithmic core; B owns evaluation infrastructure. Both
contributions (4 and 5) then build on the base, one each.

---

## 9. Evaluation Protocol

### 9.1 Metric hierarchy

**Tier 1 — PSF estimation accuracy** (only possible with synthetic/Köhler data)
- Angle error (degrees)
- Length error (pixels)
- Success rate (e.g., % of trials with angle error < 10°)

*This is the most direct measure of what you're actually improving.* Prioritize it.

**Tier 2 — Restoration quality** (needs sharp ground truth)
- PSNR (with border crop)
- SSIM (better perceptual correlation than PSNR)
- Comparison against the **true-PSF upper bound** — i.e. "how close does our estimated PSF
  get to what a perfect PSF would achieve?" **This framing is much more informative than a
  raw PSNR number** and shows sophistication.

**Tier 3 — Downstream utility** (works without sharp ground truth)
- OCR character/plate accuracy before vs. after restoration
- No-reference sharpness metrics (e.g., variance of Laplacian, gradient energy)

**Tier 4 — Qualitative**
- Side-by-side galleries
- Zoomed crops showing stroke recovery
- Documented failure cases

### 9.2 Required comparisons

Every results table should include:

| Condition | Purpose |
|---|---|
| Blurred input (do nothing) | The floor — must beat this |
| Restored with **true** PSF | The ceiling — best achievable by any estimator |
| Baseline pipeline (no pre-processing, no gate) | What the literature gives you |
| + text pre-processing | Contribution 1 isolated |
| + confidence gate | Contribution 2 isolated |
| Full pipeline | Everything together |
| Wiener instead of RL | Deconvolution-method ablation |

**The "true PSF" row is what separates a rigorous evaluation from a naive one.** It tells
you whether your remaining error is an *estimation* problem or a *deconvolution* problem.

### 9.3 Statistical hygiene

- **Multiple seeds per condition.** Report mean ± std, not single runs.
- **Fixed seeds** for reproducibility.
- **Same sweep grid** for baseline and improved versions — otherwise comparison is
  meaningless.
- **Don't tune on your test set.** Calibrate thresholds on synthetic data, report on real.

### 9.4 Figures to produce

1. Pipeline architecture diagram (§7.1)
2. Worked example panel: sharp → blurred → spectrum → cepstrum → restored
3. Angle-error heatmap: blur angle × noise level, per image type
4. Success-rate bar chart: baseline vs. each pre-processing variant
5. Confidence-vs-error scatter plot (justifies the gate threshold)
6. Confusion matrix for gating decisions
7. Before/after gallery on real plates
8. Failure-case panel

---

## 10. Risk Register & Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Angle estimation fragile on low-texture text | **High** | High | This *is* the project's target problem. Multiple scoring rules (§3.4/3.6), multiple pre-processing candidates (§7.3), semi-blind fallback (§7.5) |
| 2 | Length estimation degrades faster than angle under noise | Medium | High | Bit-plane-sliced cepstrum (Deshpande & Patnaik); confidence gate catches bad estimates |
| 3 | Wrong PSF makes output worse than input | **High** | High | Confidence gate is specifically designed for this — it's a feature, not a patch |
| 4 | FFT boundary wraparound corrupts metrics | **High** | Medium | Edge tapering + border cropping; known solved problem (§3.8) |
| 5 | Real blur is non-linear/non-uniform → poor real-world results | **High** | Medium | Expected and documented in literature (LPBlur paper). Frame proactively; synthetic + Köhler results still stand |
| 6 | Quasi-periodic text structure fakes cepstral peaks | Medium | Medium | Known theoretical concern (§2.3). Test explicitly; may motivate a pre-processing choice |
| 7 | 90° angle-convention bug | Medium | High | Verify on synthetic known-angle image in Phase 1. Cheap to catch, expensive to miss |
| 8 | Joint search too slow for live demo | Low | Low | 180 angles is sub-second per image. Only an issue for video — coarse-to-fine search if needed |
| 9 | LPBlur dataset unavailable | Medium | Low | CCPD-Blur + OCR metrics is a complete fallback |
| 10 | Scope creep (trying to handle non-uniform blur) | Medium | High | **Explicitly out of scope.** State the linear-uniform assumption in the proposal and hold the line |

### 10.1 The honest bottom line on risk

**Low-to-moderate.** The reasons:

- The two hardest mathematical components are single library calls, not from-scratch
  numerical code.
- There is a **layered fallback structure**: if automatic estimation underperforms →
  semi-blind mode; if RL rings → Wiener; if real plates fail → synthetic and Köhler results
  still constitute a complete, valid study.
- The evaluation is staged from full ground truth (synthetic) → partial (Köhler) → none
  (real plates). Even worst-case, you have rigorous quantitative results from the first two
  stages.
- The known-hard part (text-specific fragility) **is the thing you're studying**, so
  encountering it is a result, not a failure.

**The single biggest threat is scope creep**, not technical failure. Resist the urge to
handle spatially-varying blur, add deep learning, or build a full ALPR system.

---

## 11. Demo & Presentation Strategy

### 11.1 The demo that sells it

**Live, three-panel, on real images:**

```
┌──────────────┬──────────────┬──────────────┐
│   BLURRED    │  ESTIMATED   │   RESTORED   │
│    INPUT     │  PSF + conf. │    OUTPUT    │
│              │   (visual)   │              │
└──────────────┴──────────────┴──────────────┘
```

Showing the estimated PSF visually (a little line kernel image) alongside the confidence
score is what makes this feel like a *system* rather than a script. It also makes the
algorithm legible — the audience sees the machine reasoning.

**Demo sequence:**
1. A clear success — blurred plate → readable plate. Establishes it works.
2. Show the frequency-domain intermediate steps. Establishes *how* it works. **This is the
   DIP content — linger here.**
3. A case where the confidence gate refuses. Establishes sophistication — *"the system
   knows when not to act."* This is memorable and unusual.
4. A hard case, with honest commentary. Establishes credibility.

### 11.2 Slide structure

1. **The hook** — a blurred plate, unreadable. "Can we recover this without a neural
   network?"
2. **The physics** — blur as convolution; the sinc signature in the spectrum
3. **The classical pipeline** — Radon → cepstrum → RL
4. **The gap** — why text is harder than natural photos (theory + your Phase 3 numbers)
5. **Contribution 1** — text-contrast pre-processing + results table
6. **Contribution 2** — confidence gating + confusion matrix
7. **Real-world results** — CCPD/Köhler + honest limitations
8. **Live demo**
9. **Limitations & future work**

### 11.3 Presentation principles

- **Lead with the visual, follow with the math.** Show the blurred→sharp result first, then
  explain the transform-domain reasoning.
- **Show one intermediate spectrum image in full.** Nothing communicates "this is a DIP
  project" faster than a log-magnitude spectrum with visible blur stripes.
- **Include failures deliberately.** Presenting only successes reads as either lucky or
  dishonest. A documented failure case with an explanation reads as rigorous.
- **Name the assumption.** Say "we assume linear uniform blur" before anyone asks. It
  signals you understand your own method's boundaries.

### 11.4 Anticipated questions — prepare answers

| Question | Prepared answer |
|---|---|
| "Isn't this just computer vision?" | Every pipeline stage is transform-domain analysis or classical filtering. OCR appears only as an evaluation metric — remove it and the system still works identically. |
| "Why not use a CNN?" | No training data needed, fully interpretable, and the LPBlur paper documents that CNNs trained on synthetic blur fail on real plate blur anyway. |
| "What's actually novel?" | Not the algorithms — those are established. The contribution is characterizing a documented weak regime (low-texture text) and testing two targeted, classical remedies against a controlled baseline. |
| "How do you know it works?" | Ground-truth kernel validation on synthetic and Köhler data — we measure angle error in degrees and length error in pixels directly, not just eyeball outputs. |
| "What if the blur isn't linear?" | Out of scope, stated upfront. Our model is 2-parameter; real shake is a trajectory. That's the main limitation and the natural extension. |
| "Why does the confidence gate matter?" | Because deconvolution with a wrong kernel amplifies wrong frequencies — the output is worse than doing nothing. The gate converts a silent failure into an explicit "unknown." |

---

## 12. Repository Structure

```
dip-motion-deblur/
├── README.md
├── requirements.txt
├── src/
│   ├── psf.py               # kernel generation from (angle, length)
│   ├── blur.py              # synthetic blur application + noise
│   ├── spectrum.py          # FFT, log-magnitude, visualization
│   ├── radon_angle.py       # Radon-based angle scoring
│   ├── cepstrum.py          # cepstrum computation, profile, dip detection
│   ├── estimate.py          # joint (angle, length) search + confidence score
│   ├── preprocess.py        # ★ text-contrast pre-processing variants
│   ├── gate.py              # ★ confidence gating logic
│   ├── deconv.py            # RL + Wiener wrappers, edge tapering
│   ├── metrics.py           # angle/length error, PSNR, SSIM, OCR accuracy
│   └── pipeline.py          # end-to-end orchestration
├── experiments/
│   ├── run_sweep.py         # the systematic characterization sweep
│   ├── compare_preprocess.py
│   ├── calibrate_gate.py
│   └── eval_real.py
├── data/
│   ├── synthetic/
│   ├── kohler/
│   ├── ccpd_blur/
│   └── captured/
├── results/
│   ├── csv/                 # raw per-run logs
│   └── figures/
├── demo/
│   └── demo_app.py          # live three-panel demo
└── docs/
    ├── proposal.tex
    └── report.tex
```

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **PSF** | Point Spread Function — the blur kernel; how a single point of light gets smeared |
| **Blind deconvolution** | Recovering a sharp image when the blur kernel is *unknown* |
| **Non-blind deconvolution** | Recovering a sharp image when the kernel *is* known |
| **Cepstrum** | Inverse FFT of the log-magnitude spectrum; converts convolution to addition |
| **Radon transform** | Projects a 2-D image onto 1-D lines at many angles |
| **Richardson–Lucy** | Iterative maximum-likelihood deconvolution assuming Poisson noise |
| **Wiener filter** | Single-shot regularized inverse filter in the frequency domain |
| **Ringing** | Oscillating artifacts near strong edges caused by deconvolution |
| **Spatially invariant** | Same blur kernel applies everywhere in the image |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization |
| **PSNR / SSIM** | Peak Signal-to-Noise Ratio / Structural Similarity Index |

---

## 14. Full Bibliography

### Foundational
1. R. Fergus, B. Singh, A. Hertzmann, S. T. Roweis, W. T. Freeman, "Removing Camera Shake
   from a Single Photograph," *ACM TOG (SIGGRAPH)*, 25(3):787–794, 2006.
2. A. Levin, Y. Weiss, F. Durand, W. T. Freeman, "Understanding and Evaluating Blind
   Deconvolution Algorithms," *CVPR*, 2009.

### Deconvolution
3. W. H. Richardson, "Bayesian-Based Iterative Method of Image Restoration," *JOSA*,
   62(1):55–59, 1972.
4. L. B. Lucy, "An Iterative Technique for the Rectification of Observed Distributions,"
   *The Astronomical Journal*, 79(6):745–754, 1974.

### Radon / cepstral PSF estimation (core method)
5. Y. Wang et al., "A New Method and Implementation of Blind Restoration Algorithm for
   Moving Fuzzy License Plate Image Based on Frequency-Domain Characteristics," 2021.
6. X. Zhou, Y. Zhang, W. Zhou, "Improved Estimation of Motion Blur Parameters for
   Restoration from a Single Image," *PLOS ONE*, 2020.
7. V. Deshpande, M. Patnaik, "A Novel Modified Cepstral Based Technique for Blind
   Estimation of Motion Blur," *Optik*, 124(24):6994–7000, 2013.
8. "Single Image Motion Deblurring: An Accurate PSF Estimation and Ringing Reduction,"
   *Optik*, 2014.
9. W. Zhou et al., "A Self-Adaptive Learning Method for Motion Blur Kernel Estimation,"
   *Optik*, 2021.
10. "Comparative Analysis of PSF Estimation Based on Hough Transform and Radon Transform."
11. "PSF Estimation and Image Restoration for Noiseless Motion Blurred Images."

### License plates
12. "A Dataset and Model for Realistic License Plate Deblurring," arXiv:2404.13677, 2024.
13. Z. Xu et al., "Towards End-to-End License Plate Detection and Recognition: A Large
    Dataset and Baseline," *ECCV*, 2018. (CCPD)

### Benchmarks & datasets
14. R. Köhler, M. Hirsch, B. Mohler, B. Schölkopf, S. Harmeling, "Recording and Playback of
    Camera Shake: Benchmarking Blind Deconvolution with a Real-World Database," *ECCV*,
    2012.
15. S. Nah, T. H. Kim, K. M. Lee, "Deep Multi-scale Convolutional Neural Network for
    Dynamic Scene Deblurring," *CVPR*, 2017. (GoPro dataset)
16. J. Rim et al., "Real-World Blur Dataset for Learning and Benchmarking Deblurring
    Algorithms," *ECCV*, 2020. (RealBlur)

### Context (deep learning route — for related work only)
17. O. Kupyn et al., "DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial
    Networks," *CVPR*, 2018.
18. O. Kupyn et al., "DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better,"
    *ICCV*, 2019.

### Useful links
- CCPD: `https://github.com/detectRecog/CCPD`
- CCPD Zenodo mirror: `https://zenodo.org/records/15647076`
- Köhler benchmark: `https://webdav.tuebingen.mpg.de/pixel/benchmark4camerashake`
- RealBlur: `https://github.com/rimchang/RealBlur`
- Reference impl: `https://github.com/ankitVP77/Blind-Motion-Deblurring-for-Legible-License-Plates-using-Deep-Learning`
- Reference impl: `https://github.com/adamalavi/Blind-motion-deblurring-for-license-plates`
- GitHub topics: `psf-estimation`, `blind-deconvolution`, `motion-deblurring`

---

## Appendix A — First-Week Checklist

- [ ] Repo created, environment set up, dependencies installed
- [ ] Verify `skimage.transform.radon` and `skimage.restoration.richardson_lucy` import and
      run on a toy example
- [ ] Implement line-kernel PSF generator; visually inspect kernels at several angles
- [ ] Implement synthetic blur; visually confirm blur direction matches the requested angle
- [ ] **Verify the angle convention** (spectrum stripes ⊥ motion) — do not skip
- [ ] Deconvolve with the true PSF; confirm PSNR improves after border cropping
- [ ] Download CCPD (at minimum `ccpd_blur` and a slice of `ccpd_base`)
- [ ] Check whether LPBlur data is publicly downloadable
- [ ] Download Köhler benchmark
- [ ] Take 10–20 of your own motion-blurred text/plate photos for the demo

## Appendix B — Things That Will Waste Your Time If You're Not Careful

1. **The 90° convention bug.** Spectrum stripes are perpendicular to motion. Verify early.
2. **Boundary artifacts corrupting metrics.** Your pipeline may work while your numbers say
   it doesn't. Always inspect the actual output image, not just the PSNR.
3. **Testing on one lucky configuration.** One angle, one length, one image proves nothing.
   Sweep or don't bother.
4. **Non-normalized kernels.** If the PSF doesn't sum to 1, brightness shifts and metrics
   lie.
5. **Tuning on the test set.** Calibrate on synthetic, report on real.
6. **Scope creep into non-uniform blur.** It's a different, much harder problem. Say no.
7. **Not saving experiment configs.** A results number with no recorded config is useless
   in three weeks.
