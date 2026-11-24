# Progress log — Track A: NMF note transcription

*EEP55C36 final project · Smaragdis & Brown (2003) · individual project*

*Deadline: 2026-11-24*

This file records decisions, problems and next steps as they happen.

---

## 1. Project snapshot

| Item                                 | Value                                                                 |
|--------------------------------------|-----------------------------------------------------------------------|
| Track                                | A — Audio, generalized-KL NMF of magnitude spectrograms               |
| Baseline                             | Multiplicative KL updates, frozen-dictionary inference                |
| Modification                         | Temporal smoothness penalty on activations, weight $\mu$              |
| Primary metric (declared in advance) | Frame-level note $F_1$ on test clips, larger is better                |
| Secondary metrics                    | $D/(FT)$ (unpenalized KL), onset smearing, runtime                    |
| Python / OS tested                   | Python 3.13.18, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2, pytest 9.2.0; macOS 27.2.1, Apple M4 (arm64) |
| Repository URL                       | https://github.com/jakovkacan/cma-project-audio                       |
| Baseline tag / hash                  | `v0.1-baseline` / _TODO — tag not created yet_                        |
| Submission tag / full hash           | `v1.0-submission` / _TODO_                                            |

## 2. Scope statement

- **Reproduced:** generalized-KL NMF, $V \approx WH$, multiplicative updates, templates in columns of $W$, activations
  in rows of $H$.
- **Added protocol (not in the paper):** dictionary learned on development clips, frozen, activations inferred on
  held-out clips.
- **Also demonstrated:** blind joint factorization on a single clip (coinciding-notes merge).
- **Not reproduced:** the paper's real piano recording, Euclidean-cost experiments, energy-penalty variant (Eq. 4).
- **Paper quirks noted:** printed gradient labels in Eqs. (5)–(6) are dimensionally swapped; use explicit KL updates
  (Eq. 7 form).

## 3. Fixed protocol choices

Fill in and freeze **before** looking at test results.

| Choice                                               | Value                                                                                                   | Decided on |
|------------------------------------------------------|---------------------------------------------------------------------------------------------------------|------------|
| Number of components $K$                             | 3                                                                                                       | 2026-11-24 |
| STFT                                                 | Hann, `nperseg=512`, `noverlap=384` (hop 128), `fs=8000`, $F=257$                                       | 2026-11-13 |
| Magnitude normalization                              | Baseline: none (raw magnitudes). Modification: _TODO — one constant from development data, or $\mu=\tilde\mu/\text{mean}(V)$_ |            |
| Initialization                                       | $W_0, H_0 \sim \mathcal{U}[0.5, 1.5)$ from `default_rng(seed)`; dictionary restart seeds 0–3; inference seed = clip seed, identical for both methods | 2026-11-14 |
| Iterations (both methods)                            | 300 for dictionary learning, 200 for frozen inference                                                   | 2026-11-24 |
| `eps` floor                                          | $10^{-10}$                                                                                              | 2026-11-14 |
| Development seeds                                    | 0–9, at `dev_noise_std = 0.005`                                                                         | 2026-11-24 |
| Validation seeds                                     | 100–109                                                                                                 | 2026-11-24 |
| Test seeds                                           | 1000–1029 (30 cases per setting)                                                                        | 2026-11-24 |
| Difficulty settings                                  | Noise std 0.05 / 0.10 / 0.20 (easy / medium / hard); see decision log                                   | 2026-11-24 |
| Threshold grid (both methods)                        | 0.05, 0.10, …, 0.95; re-selected per noise level on validation, ties to the smallest $\theta$           | 2026-11-24 |
| $\mu$ grid (modified only)                           | _TODO_                                                                                                  |            |
| Baseline extra search budget (to match $\mu$ tuning) | _TODO_                                                                                                  |            |
| Component-to-pitch matching                          | Cosine similarity to exclusive-frame reference templates + Hungarian assignment, development data only  | 2026-11-14 |
| Scale fixing                                         | $c_k$ = 95th percentile of development activations over frames where pitch $k$ sounds                   | 2026-11-14 |
| Uncertainty                                          | Mean with Student-t 95% interval over clips (`src/stats.py::mean_ci`); applied to paired differences once the modification exists | 2026-11-24 |

Dates are those of the commits that introduced each choice into the code or configs.

## 4. Objective and modification (to fill in as derived)

Baseline objective:

$$D(V\,\|\,WH)=\sum_{f,t}\left(V_{ft}\log\frac{V_{ft}}{(WH)_{ft}}-V_{ft}+(WH)_{ft}\right)$$

Modified objective:

$$J(W,H)=D(V\,\|\,WH)+\frac{\mu}{2}\sum_{k}\sum_{t=2}^{T}\left(H_{kt}-H_{k,t-1}\right)^2$$

- **Update rule derived:** _TODO — positive/negative gradient split, boundary frames_
- **Monotonicity check:** _TODO — does $J$ decrease every iteration? If not, note and decide on the quadratic-root form_
- **Prior work cited:** _TODO — Virtanen (2007); verify citation details_
- **Hypothesis (written before experiments):**
    - Helps when: _TODO_
    - Hurts when: _TODO_
- **Extra cost introduced:** _TODO (expected small, $O(KT)$ per iteration)_

## 5. Verification checklist

- [ ] Hand-computable tiny example (one update step checked on paper) — hand-computed values are covered in tests
  (`test_hand_activation_update`, `test_hand_template_update`, `test_hand_example`); _paper check to confirm_
- [x] Limiting cases: $K=1$ (`test_rank_one_closed_form`), exact rank-$K$ input (`test_exact_low_rank_recovery`),
  zero column in $V$ (`test_zero_frame_and_zero_bin`, `test_zero_column_is_safe`). Single-frame input ($T=1$)
  has no test; checked by hand on 2026-11-24: `fit_nmf` and `infer_activations` return finite, non-negative factors
  with the right shapes, and the loss is non-increasing.
- [x] Factors non-negative and finite; shapes correct (`test_nonnegative_shapes_and_finite_with_zeros`,
  `test_factors_valid`)
- [x] Baseline KL non-increasing (`test_single_updates_do_not_increase_kl`, `test_monotone_on_random_data`,
  `test_monotone_on_clip`)
- [x] Inputs not mutated (`test_inputs_not_mutated` and similar, in every module)
- [x] Template normalization leaves $WH$ unchanged (`test_product_unchanged_and_columns_sum_to_one`)
- [ ] Constant-in-time $H$ gives zero penalty
- [ ] Modified with $\mu=0$ equals baseline
- [ ] Optional cross-check against an external KL-NMF (validation only, cited, kept separate)

Full suite on 2026-11-24: 203 passed, 1 skipped (the optional `requirements-dev.txt` check), about 40 s.

## 6. Experiment checklist

- [ ] Smoke config runs in about a minute and shows both methods — runs in about 3 s, baseline only so far
- [ ] Final config reproduces all tables and figures — `configs/baseline.json` reproduces the baseline tables and
  figures (about 40 s); `configs/final.json` not written yet
- [ ] Three difficulty settings, ≥ 20 paired clips each — three settings × 30 test clips for the baseline; pairing
  with the modified method still to come
- [ ] Ablation: $\mu=0$ with threshold re-selected
- [ ] Extra control: baseline plus post-hoc smoothing filter
- [ ] Stress: shorter notes
- [ ] Stress: changed harmonic profile
- [ ] Stress: notes that always coincide in learning data
- [ ] Blind joint-factorization demo on one clip — checked numerically by `test_blind_factorization_of_clip0`; no
  figure or write-up yet
- [x] Timing: warm-up, repeats, setup reported separately, hardware recorded — 1 warm-up + 5 repeats per clip,
  dictionary setup time in `run_info.json`, BLAS pinned to one thread, platform and versions in `run_info.json`
- [ ] Figures: representative example, overall paired comparison, stress case — `fig_example.png`,
  `fig_templates.png` and `fig_f1_by_noise.png` exist for the baseline; paired comparison and stress figures to come

## 7. Decision log

*Newest first. One entry per decision.*

### 2026-11-24 — One shared dictionary across noise levels

- **Context:** the test settings differ only in noise level. The dictionary could be learned once or re-learned at
  each noise level.
- **Options considered:** (a) learn $W$ and $c_k$ once at the development noise (0.005) and reuse them everywhere;
  (b) learn a separate dictionary per noise level.
- **Decision:** (a). $\theta$ is still re-selected per noise level on validation clips.
- **Why:** holding the templates fixed means differences between methods and settings come from activation
  inference alone, which is exactly where the smoothness penalty acts. It also keeps the matching and scaling
  computed once, on development data only.
- **Consequences / follow-ups:** baseline $F_1$ at high noise includes a train/test noise mismatch. This is
  intended, but the report must state it. Implemented in `run_experiments.py` (`learn_dictionary` once, then
  `evaluate_setting` per noise level).

### 2026-11-24 — Noise calibration: σ ∈ {0.05, 0.10, 0.20}

- **Context:** the difficulty settings need a useful spread of baseline $F_1$ and must be chosen without test data.
- **Options considered:** noise std 0.005, 0.02, 0.05, 0.10, 0.20.
- **Decision:** test settings 0.05 / 0.10 / 0.20; development noise 0.005.
- **Why:** with $\theta$ chosen on validation seeds 100–104 and scored on validation seeds 105–109 (dictionary from
  dev seeds 0–9, 4 restarts, 300 + 200 iterations):

  | noise std          | 0.005         | 0.02          | 0.05          | 0.10          | 0.20          |
  |--------------------|---------------|---------------|---------------|---------------|---------------|
  | selected $\theta$  | 0.35          | 0.35          | 0.40          | 0.55          | 0.35          |
  | val $F_1$ (mean ± sd) | 0.994 ± 0.003 | 0.994 ± 0.003 | 0.991 ± 0.001 | 0.938 ± 0.026 | 0.544 ± 0.020 |

  0.005 and 0.02 are indistinguishable from each other and at ceiling. 0.05 / 0.10 / 0.20 span near-ceiling,
  moderate and clearly degraded performance. The numbers match the guide's reference run exactly.
- **Consequences / follow-ups:** frozen in `configs/baseline.json` (`noise_levels`, `dev_noise_std`). No test seed
  was used in this calibration.

### 2026-11-24 — Four restarts for the dictionary

- **Context:** KL-NMF is non-convex. A single initialization can converge to a poor local minimum.
- **Options considered:** a single seed; best of several restarts by final loss.
- **Decision:** restarts with seeds 0–3 on the concatenated development clips, keeping the lowest final loss.
- **Why:** measured on dev seeds 0–9 with 300 iterations, the final loss per entry is $8.25\times10^{-4}$ for
  seed 0 and $3.764\times10^{-4}$ for seeds 1–3 (best seed 3; seeds 1–3 agree to about five significant figures).
  With seed 0 alone, the matched similarities are [0.761, 1.000, 0.162]: a merged A3/E4 template plus a noise
  template. The best restart gives [0.9998, 0.9998, 0.9991], peak bins 218.75 / 281.25 / 328.125 Hz,
  exclusive-frame counts [427, 486, 438] and scales $c \approx$ [2.14, 2.18, 2.17].
- **Consequences / follow-ups:** dictionary setup takes about 19 s of the 40 s baseline run. Restart seeds are a
  config key (`restart_seeds`).

## 8. Problems and fixes

| Date       | Problem                                                           | Cause                | Fix                                         | Commit    |
|------------|-------------------------------------------------------------------|----------------------|---------------------------------------------|-----------|
| 2026-11-14 | `concat_clips` accepted clips whose $V$ and $Y$ lengths differed  | No length check      | Check that frame counts agree; raise otherwise | `56ba539` |
| 2026-11-14 | `split_seeds` accepted empty and overlapping seed ranges          | No validation        | Raise `ValueError` for empty or overlapping ranges | `7a25ec4` |
| 2026-11-14 | `make_clip` rejected `noise_std = 0`                              | Bound was strict     | Allow `noise_std = 0`                       | `7a25ec4` |

## 9. Session notes

*Short dated entries: what was done, what is next.*

### 2026-11-13

- Read paper, brief and Track A guide.
- Drafted project structure and plan.
- Create repository, commit plan.
- Implement `data.py` generator.

### 2026-11-14

- Added full test suite for all modules.
- Implemented `baseline.py` and `alignment.py`.

### 2026-11-16

- Partial `metrics.py` implementation.

### 2026-11-24

- Completed `metrics.py` implementation.
- Implemented `protocol.py`, `stats.py`, `timing.py`, `run_experiments.py`
- Added `smoke` and `baseline` configs
- Added `make_figures.py`; both configs run end to end and produce the three figures.
- Re-ran the restart and noise-calibration checks; recorded them in the decision log.
- Wrote `README.md`.
- **Next:** commit `make_figures.py`, tag `v0.1-baseline` and record its hash; then start `modified.py`.

## 10. Open questions

- [ ] Is the simple split-gradient $H$ update monotone, or is the exact quadratic form needed?
- [x] Which noise levels give a useful spread of baseline $F_1$? — 0.05 / 0.10 / 0.20 (see decision log).
- [ ] The selected $\theta$ is not monotone in noise (0.40, 0.55, 0.35). Why does it drop at 0.20? Look at the
  validation $F_1$ curves in `run_info.json["validation"]`.

## 11. Results summary

*Fill after the final run. State the primary-metric result honestly, including a null or negative one.*

Baseline only so far (`configs/baseline.json`, 30 test clips per setting, mean and 95% interval):

| Noise std | $\theta$ | Frame $F_1$          | $D/(FT)$ | Mean \|onset lag\| (frames) | Onsets missed |
|-----------|----------|----------------------|----------|-----------------------------|---------------|
| 0.05      | 0.40     | 0.988 [0.985, 0.990] | 0.00270  | 0.12 [0.07, 0.16]           | 0             |
| 0.10      | 0.55     | 0.933 [0.925, 0.941] | 0.00579  | 0.46 [0.39, 0.54]           | 0             |
| 0.20      | 0.35     | 0.545 [0.536, 0.554] | 0.01233  | 2.69 [2.67, 2.71]           | 0             |

- **Primary metric (mean paired difference, 95% CI) per setting:** _TODO — needs the modified method_
- **Reconstruction loss $D/(FT)$:** baseline in the table above
- **Onset smearing:** baseline in the table above; lag grows sharply at noise 0.20
- **Runtime overhead:** baseline inference about 35 ms per clip (200 iterations, one BLAS thread); dictionary
  setup about 19 s. Overhead of the modification _TODO_
- **Ablation:** _TODO_
- **Stress cases:** _TODO_
- **Least convincing result / known limitations:** baseline $F_1$ falls to 0.545 at noise 0.20, so that setting
  has the most room for the modification but also the most train/test noise mismatch (dictionary learned at 0.005).

## 12. Pre-submission checklist

- [ ] README works from a clean clone: install, smoke run, tests, final run — commands verified in the working
  environment, not yet from a fresh clone
- [ ] Report claims match saved results
- [ ] No test labels used in alignment, scaling or tuning — enforced for the baseline by
  `test_uses_only_development_clips`, `test_no_development_seeds_and_noise_level_used` and
  `test_threshold_independent_of_test_clips`; recheck for the modification
- [ ] Seeds, versions, hardware recorded in `results/environment.json` — recorded in `run_info.json`
  (`config`, `environment`) instead; `platform.processor()` reports only `arm`, so state the CPU in the report
- [ ] No paper PDFs, `.venv`, or secrets committed — none tracked as of `e77deb9`, but `docs/baseline_guide.pdf`,
  `.DS_Store` and `__pycache__/` are untracked and not in `.gitignore`
- [ ] Teaching team has repository access
- [ ] Tag `v1.0-submission` created; full hash recorded in README, report and Section 1
- [ ] Assistance and attribution declaration written
- [ ] Interview backup (saved figures or screenshots) ready

## 13. Attribution and assistance

- **Paper:** P. Smaragdis and J. C. Brown (2003), WASPAA, 177–180.
- **External methods / code cited:**
    - D. D. Lee and H. S. Seung (2001), "Algorithms for non-negative matrix factorization", NIPS 13 —
      multiplicative KL updates.
    - `scipy.signal.stft` (spectrograms), `scipy.optimize.linear_sum_assignment` (Hungarian matching).
- **Collaboration:** _TODO_
- **AI assistance:** Claude Code
    - `progress.md` progress skeleton
    - Docstring generation
    - Unit tests `test_data.py`; full test suite (`7824e71`)
    - `reference_templates` in `alignment.py` and `select_threshold` in `metrics.py` (marked in the code)
    - fixes for `onset_lags`, `select_threshold`, `time_call`
    - `run_experiments.py`
    - `README.md` and filling in this log from the code, configs and saved results
