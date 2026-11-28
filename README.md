# NMF note transcription: KL-NMF baseline (Track A)

EEP55C36 final project. Reproduction of the non-negative matrix factorization approach to polyphonic
transcription of Smaragdis & Brown (2003), on synthetic harmonic audio.

A magnitude spectrogram $V$ (shape $F \times T$) is factorized as $V \approx WH$ by minimizing the generalized
Kullback–Leibler divergence with multiplicative updates. Columns of $W$ are spectral templates; rows of $H$ are
their activations over time. A dictionary is learned on development clips, aligned to pitches, frozen, and used to
infer activations on held-out clips, which are thresholded into note predictions and scored against the
ground-truth note activity.

**Modification:** a temporal smoothness penalty on the activations (Virtanen, 2007), $\frac{\mu}{2}\sum_{k,t}
(H_{kt}-H_{k,t-1})^2$, added to the KL objective during frozen-dictionary inference (`src/modified.py`). It is
compared with the baseline, a $\mu = 0$ ablation, and two post-hoc smoothing controls (moving average and running
median, `src/postprocess.py`), on three noise levels and two stress cases.

**Status:** baseline complete and tagged (`v0.1-baseline`); modification, control, stress cases and figures
implemented. Main finding: the penalty does not improve frame $F_1$ on the main settings; see
[`docs/report_notes.md`](docs/report_notes.md) for the hypothesis, results and interpretation.

## Installation

Tested with Python 3.13.18 on macOS 27.2 (Apple M4, arm64). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins `numpy`, `scipy`, `matplotlib` and `pytest`.

## Quick start

All commands run from the repository root.

```bash
python -m pytest                                       # checks, ~1 min
python run_experiments.py --config configs/smoke.json  # smoke run, both methods, ~5 s
python make_figures.py --results results/smoke         # figures, ~1 s
```

## Checks

```bash
python -m pytest                    # full suite, ~1 min (323 passed, 1 skipped)
python -m pytest -m "not slow"      # quick loop
python -m pytest tests/test_baseline.py
python -m pytest -rs                # list skipped tests and why
```

The one skipped test checks the optional `requirements-dev.txt`, which this repository does not use (`pytest` is
pinned in `requirements.txt`). See [`tests/README.md`](tests/README.md) for what each test file covers, including
the hand-computed update examples, monotonicity checks and data-leakage checks on the protocol.

## Running the experiments

| Config                  | Dev / val / test clips | Noise levels      | Restarts | Run time |
|-------------------------|------------------------|-------------------|----------|----------|
| `configs/smoke.json`    | 3 / 3 / 5              | 0.05, 0.20        | 2        | ~5 s     |
| `configs/baseline.json` | 10 / 10 / 30           | 0.05, 0.10, 0.20  | 4        | ~40 s    |
| `configs/final.json`    | 10 / 10 / 30           | 0.05, 0.10, 0.20  | 4        | ~2.5 min |
| `configs/stress_short.json`  | 10 / 10 / 30      | 0.05, 0.10, 0.20  | 4        | ~2 min   |
| `configs/stress_timbre.json` | 10 / 10 / 30      | 0.05, 0.10, 0.20  | 4        | ~2 min   |
| `configs/final_wide.json`, `stress_short_wide.json`, `stress_timbre_wide.json` | 10 / 10 / 30 | 0.05, 0.10, 0.20 | 4 | ~2.5 min each |

**Methods.** Each config's optional `methods` list names the methods to run on identical clips:

| Label            | What it runs                                                                 | Also tuned on validation |
|------------------|------------------------------------------------------------------------------|--------------------------|
| `baseline`       | KL inference with frozen $W$ (no grid in the entry)                          | —                        |
| `smooth`         | smoothness-penalized inference (`"mu_grid"`)                                 | $\mu$                    |
| `smooth_mu0`     | the penalized code path with `"mu_grid": [0.0]` (ablation)                   | —                        |
| `posthoc_mean`   | baseline, then a moving average of $H$ (`"window_grid"`, `"filter": "mean"`) | window                   |
| `posthoc_median` | baseline, then a running median of $H$ (`"filter": "median"`)               | window                   |

`smoke.json` runs `baseline` and `smooth`; `baseline.json` the baseline only; `final.json` all five. The stress
configs keep `final.json`'s protocol and methods (without the ablation) and change only the validation and test clips
through `"clip_kwargs"`: notes of 0.10–0.20 s (`stress_short`) or harmonic amplitudes $1/h^2$ (`stress_timbre`).
The `*_wide` configs repeat `final`, `stress_short` and `stress_timbre` with the threshold grid widened to 3.0: a
sensitivity check, added because at σ = 0.20 the best threshold can lie above the standard grid's top value of 0.95.

```bash
python run_experiments.py --config configs/baseline.json
python make_figures.py --results results/baseline
python run_experiments.py --config configs/final.json      # baseline vs modified vs controls
python make_figures.py --results results/final
for c in stress_short stress_timbre final_wide stress_short_wide stress_timbre_wide; do
  python run_experiments.py --config configs/$c.json && python make_figures.py --results results/$c
done
```

Times are wall-clock on an Apple silicon laptop with BLAS pinned to one thread (`run_experiments.py` sets
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1 unless they are already set). Learning the
dictionary takes about 19 s of the baseline run.

### What a run does

1. **Dictionary.** Generate the development clips at `dev_noise_std`, concatenate them, and fit KL-NMF from each
   seed in `restart_seeds`, keeping the lowest final loss.
2. **Alignment.** Build a reference template per pitch from frames where only that pitch sounds, match components
   to pitches by cosine similarity (Hungarian assignment), and fix per-pitch activation scales from development
   activations. Only development data is used.
3. **Threshold.** For each noise level and method, infer activations on validation clips with $W$ frozen and choose
   the threshold $\theta$ that maximizes mean per-clip $F_1$. For the modified method, the penalty weight $\mu$ is
   chosen at the same time: the best $\theta$ is found for each $\mu$ in `mu_grid`, and the $\mu$ with the highest
   validation $F_1$ wins (ties go to the smallest $\mu$).
4. **Test.** Infer activations on test clips and score them. Test labels are used only here.

### Outputs

Each run writes to the `output_dir` set in its config (`results/<name>/`, ignored by git):

| File                | Contents                                                                                       |
|---------------------|------------------------------------------------------------------------------------------------|
| `cases.csv`         | One row per (test clip, metric): `case_id, split, seed, noise_std, method, metric, value, runtime_s` |
| `summary.csv`       | One row per (noise level, method, metric): mean, 95% Student-t interval over clips, sd, n      |
| `paired.csv`        | Only when `baseline` and another method ran: per (noise level, method, metric, plus `runtime_s`), mean paired difference method − baseline over test clips with its 95% interval |
| `run_info.json`     | Full config, methods, selected thresholds and $\mu$, validation curves (per $\mu$ for the modified method), best restart, match similarities, scales, setup time, software versions, platform, thread settings |
| `example_clip.npz`  | One fixed validation clip with $V$, $Y$, scaled $H$, $\theta$, aligned $W$ and references $R$  |
| `figures/fig_example.png`     | Spectrogram, true note activity and scaled activations with $\theta$                 |
| `figures/fig_templates.png`   | Learned templates against the reference templates, one panel per pitch               |
| `figures/fig_f1_by_noise.png` | Per-clip test $F_1$ with mean and 95% interval, per noise level and method           |
| `figures/fig_paired.png`      | Per-clip differences to the baseline ($F_1$ and onset error) with mean and 95% interval (only with `paired.csv`) |
| `figures/fig_validation.png`  | Best validation $F_1$ against $\mu$ or window, per noise level, with the baseline as a dashed line (only for tuned methods) |
| `figures/fig_example_methods.png` | The example clip's scaled activations for every method that changes them      |

`make_figures.py` reads only these saved files; it never refits or reruns inference.

### Metrics

- **Frame-level $F_1$** (primary): over all (pitch, frame) entries of a clip, prediction $H_{\text{scaled}} > \theta$.
- **`kl_per_entry`**: reconstruction loss $D(V\,\|\,WH)/(FT)$.
- **`onset_mean_abs_lag_frames`, `onsets_missed`**: onset timing error within ±`onset_max_lag` frames.
- **`runtime_s`**: inference time per clip.

The clip is the unit of analysis: intervals are computed over one value per clip, never over frames.

## Baseline results

From `configs/baseline.json`, 30 test clips per noise level (mean and 95% interval):

| Noise std | $\theta$ | Frame $F_1$           | $D/(FT)$ | Mean onset lag (frames) |
|-----------|----------|-----------------------|----------|-------------------------|
| 0.05      | 0.40     | 0.988 [0.985, 0.990]  | 0.0027   | 0.12                    |
| 0.10      | 0.55     | 0.933 [0.925, 0.941]  | 0.0058   | 0.46                    |
| 0.20      | 0.35     | 0.545 [0.536, 0.554]  | 0.0123   | 2.69                    |

Matched template similarities are above 0.999 for all three pitches. Rerunning the config reproduces these numbers.

The protocol choices behind these numbers are recorded with their evidence in
[`docs/progress.md`](docs/progress.md) (Sections 3 and 7):

- **Restarts:** a single initialization (seed 0) merges A3 and E4 into one template and learns a noise template.
  Four restarts avoid this.
- **Noise levels:** 0.05, 0.10 and 0.20 were calibrated on validation clips only.
- **Shared dictionary:** one dictionary is learned at the development noise and reused at every level, so
  comparisons isolate activation inference.

## Data

`src/data.py` synthesizes 6 s clips at 8 kHz containing three harmonic tones (A3, C♯4, E4), each with six
harmonics of amplitude $1/h$, random note timings and durations of 0.35–0.65 s, plus Gaussian noise of a chosen
standard deviation. Spectrograms use a Hann STFT with `nperseg=512` and `noverlap=384` ($F = 257$). Each clip is
fully determined by its seed; development, validation and test seeds come from disjoint ranges set in the config.
Noise is drawn independently of the notes, so the same seed gives the same notes at every noise level.

## Repository layout

```
├── run_experiments.py   config -> protocol -> CSV + JSON
├── make_figures.py      saved results -> PNG figures
├── configs/             smoke, baseline, final, stress_short, stress_timbre, stress_short_wide
├── src/
│   ├── data.py          synthetic clips (with stress options), seed splits, concatenation
│   ├── baseline.py      KL divergence, multiplicative updates, fit with restarts, frozen inference
│   ├── modified.py      temporal smoothness penalty, penalized H update, frozen inference
│   ├── postprocess.py   post-hoc moving-average / running-median smoothing (control)
│   ├── alignment.py     reference templates, component matching, activation scales
│   ├── metrics.py       frame F1, KL per entry, onset lags, threshold selection
│   ├── protocol.py      learn_dictionary, evaluate_setting (baseline / modified / control)
│   ├── stats.py         mean with Student-t interval
│   └── timing.py        warm-up + repeated timing
├── tests/               pytest suite (see tests/README.md)
└── docs/
    ├── progress.md      decisions, problems and progress log
    └── report_notes.md  hypothesis, results and interpretation, drafted for the report
```

## Scope

- **Reproduced:** generalized-KL NMF with multiplicative updates; templates in $W$, activations in $H$.
- **Added (not in the paper):** dictionary learned on development clips, frozen, activations inferred on held-out
  clips; validation-selected threshold; paired per-clip evaluation.
- **Modification (not in the paper):** temporal smoothness penalty on $H$ during inference, after Virtanen (2007);
  post-hoc smoothing filters as a control.
- **Not reproduced:** the paper's real piano recording, the Euclidean-cost experiments and the energy-penalty
  variant.
- The printed gradient labels in the paper's Eqs. (5)–(6) are dimensionally swapped; the code uses the explicit KL
  update form.

## References and attribution

- P. Smaragdis and J. C. Brown, "Non-negative matrix factorization for polyphonic music transcription," *IEEE
  WASPAA*, 2003, pp. 177–180.
- D. D. Lee and H. S. Seung, "Algorithms for non-negative matrix factorization," *NIPS*, 2001.
- T. Virtanen, "Monaural sound source separation by nonnegative matrix factorization with temporal continuity and
  sparseness criteria," *IEEE TASLP*, 15(3), 1066–1074, 2007 (the smoothness penalty).
- SciPy: `scipy.ndimage.uniform_filter1d` and `median_filter` for the post-hoc control.
- SciPy: `scipy.signal.stft` for spectrograms, `scipy.optimize.linear_sum_assignment` for component matching.
- AI assistance (Claude Code) was used for the test suite, docstrings, `reference_templates`, `select_threshold`,
  fixes in `metrics.py` and `timing.py`, `run_experiments.py`, this README, and in the modification milestone for
  `src/modified.py`, `src/postprocess.py`, the protocol and runner extensions, the stress options in `src/data.py`,
  the new figures in `make_figures.py` and `docs/report_notes.md`. Details are in
  [`docs/progress.md`](docs/progress.md), Section 13.
