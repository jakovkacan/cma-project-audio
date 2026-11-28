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
| Baseline tag / hash                  | `v0.1-baseline` / tag `8ed44d4e772439dbc8727943ca60af1628660545` (`git rev-parse v0.1-baseline`), commit `3e68fd954a6756b7fea799edb271495a7da345be` |
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
| Magnitude normalization                              | None for either method (raw magnitudes, raw $\mu$). $\mu$ is re-selected per noise level on validation clips, which absorbs the change in the scale of $H$ with noise (see decision log) | 2026-11-28 |
| Initialization                                       | $W_0, H_0 \sim \mathcal{U}[0.5, 1.5)$ from `default_rng(seed)`; dictionary restart seeds 0–3; inference seed = clip seed, identical for both methods | 2026-11-14 |
| Iterations (both methods)                            | 300 for dictionary learning, 200 for frozen inference                                                   | 2026-11-24 |
| `eps` floor                                          | $10^{-10}$                                                                                              | 2026-11-14 |
| Development seeds                                    | 0–9, at `dev_noise_std = 0.005`                                                                         | 2026-11-24 |
| Validation seeds                                     | 100–109                                                                                                 | 2026-11-24 |
| Test seeds                                           | 1000–1029 (30 cases per setting)                                                                        | 2026-11-24 |
| Difficulty settings                                  | Noise std 0.05 / 0.10 / 0.20 (easy / medium / hard); see decision log                                   | 2026-11-24 |
| Threshold grid (both methods)                        | 0.05, 0.10, …, 0.95; re-selected per noise level on validation, ties to the smallest $\theta$           | 2026-11-24 |
| $\mu$ grid (modified only)                           | 0.3, 1, 3, 10, 30, 100; selected jointly with $\theta$ per noise level on validation, ties to the smallest $\mu$. $\mu=0$ is excluded (the method always smooths); it runs separately as the ablation `smooth_mu0`. Chosen from validation clip 100 only | 2026-11-28 |
| Baseline extra search budget (to match $\mu$ tuning) | _TODO — modified searches 6 × 19 = 114 ($\mu$, $\theta$) pairs, baseline 19. Option: a 6× finer $\theta$ grid for the baseline. Less pressing now that the modified method does not win (see Section 11)_ |            |
| Component-to-pitch matching                          | Cosine similarity to exclusive-frame reference templates + Hungarian assignment, development data only  | 2026-11-14 |
| Scale fixing                                         | $c_k$ = 95th percentile of development activations over frames where pitch $k$ sounds                   | 2026-11-14 |
| Uncertainty                                          | Mean with Student-t 95% interval over clips (`src/stats.py::mean_ci`); applied to paired differences once the modification exists | 2026-11-24 |

Dates are those of the commits that introduced each choice into the code or configs.

## 4. Objective and modification (to fill in as derived)

Baseline objective:

$$D(V\,\|\,WH)=\sum_{f,t}\left(V_{ft}\log\frac{V_{ft}}{(WH)_{ft}}-V_{ft}+(WH)_{ft}\right)$$

Modified objective:

$$J(W,H)=D(V\,\|\,WH)+\frac{\mu}{2}\sum_{k}\sum_{t=2}^{T}\left(H_{kt}-H_{k,t-1}\right)^2$$

- **Update rule derived (2026-11-28, `src/modified.py::update_activations_smooth`):** the penalty gradient is
  $\partial P/\partial H_{kt}=\mu(c_tH_{kt}-N_{kt})$ with $N_{kt}=H_{k,t-1}+H_{k,t+1}$ (existing neighbours only) and
  $c_t$ the neighbour count (2 inside, 1 at the first and last frame, 0 if $T=1$). Splitting the gradient of $J$ into
  positive and negative parts gives
  $$H_{kt}\leftarrow H_{kt}\,\frac{\sum_f W_{fk}Q_{ft}+\mu N_{kt}}{\sum_f W_{fk}+\mu c_tH_{kt}}.$$
  With $\mu=0$ this is the baseline update bit for bit. Extra work is $O(KT)$ per iteration. Only inference changes;
  $W$ comes from the baseline `learn_dictionary` and stays frozen, and the initial $H$ is the baseline's for the same seed.
- **Monotonicity check (2026-11-28):** the gradient-split rule has no descent proof, so it was checked empirically.
  With the smoke dictionary on validation clip 100 at noise 0.05 and 0.20, $J$ never increased over 500 iterations for
  $\mu\in\{0.01,0.1,1,10,100,1000\}$; the tests also check it on random data and on clip 0
  (`test_objective_monotone_on_*`). The quadratic-root form is not needed for now.
  **Caveat:** for $\mu\ge100$, $J$ is still falling at 200 iterations (the `n_iter_infer` budget), because the random
  start is far from smooth. Large-$\mu$ results at a fixed budget partly reflect incomplete convergence, so record this
  when choosing the $\mu$ grid. On these clips mean $H\approx0.5$–$1$; the penalty barely changes $J$ for
  $\mu\le0.1$, so a useful grid probably lies around $\mu\in[1,100]$ (to be selected on validation clips).
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
- [x] Constant-in-time $H$ gives zero penalty (`test_constant_in_time_is_zero`, `test_constant_activations_recovered`)
- [x] Modified with $\mu=0$ equals baseline (`test_mu_zero_is_baseline_exactly`, update and inference, bitwise)
- [ ] Optional cross-check against an external KL-NMF (validation only, cited, kept separate)

Full suite on 2026-11-24: 203 passed, 1 skipped (the optional `requirements-dev.txt` check), about 40 s.

## 6. Experiment checklist

- [x] Smoke config runs in about a minute and shows both methods — about 5 s, `baseline` and `smooth` ($\mu$ grid 1, 10, 100)
- [ ] Final config reproduces all tables and figures — `configs/final.json` runs in about 90 s and writes
  `cases.csv`, `summary.csv`, `paired.csv`, `run_info.json`, `example_clip.npz`; paired-comparison figure still to come
- [x] Three difficulty settings, ≥ 20 paired clips each — three settings × 30 test clips, paired by seed;
  differences in `paired.csv`
- [x] Ablation: $\mu=0$ with threshold re-selected — `smooth_mu0` in `configs/final.json`; every quality difference
  to the baseline is exactly 0 (checked by `test_ablation_differences_are_zero`)
- [x] Extra control: baseline plus post-hoc smoothing filter — `posthoc_mean` and `posthoc_median` in `final.json`
- [x] Stress: shorter notes — `configs/stress_short.json` (+ `stress_short_wide.json` sensitivity check)
- [x] Stress: changed harmonic profile — `configs/stress_timbre.json` ($1/h^2$)
- [ ] Stress: notes that always coincide in learning data — not runnable through the protocol (no single-pitch
  frames for alignment); see decision log 2026-11-28; covered by the blind joint factorization
- [ ] Blind joint-factorization demo on one clip — checked numerically by `test_blind_factorization_of_clip0`; no
  figure or write-up yet
- [x] Timing: warm-up, repeats, setup reported separately, hardware recorded — 1 warm-up + 5 repeats per clip,
  dictionary setup time in `run_info.json`, BLAS pinned to one thread, platform and versions in `run_info.json`
- [x] Figures: representative example, overall paired comparison, stress case — `fig_example_methods.png`,
  `fig_paired.png` (final and each stress run), `fig_validation.png`, plus the baseline figures

## 7. Decision log

*Newest first. One entry per decision.*

### 2026-11-28 — Wider threshold grid as a sensitivity check for short notes

- **Context:** in `stress_short` at σ = 0.20 the baseline (and both post-hoc controls) selected θ = 0.95, the top
  of the grid. Validation $F_1$ was still rising there (0.240 at 0.90, 0.293 at 0.95), so the baseline was limited by
  the grid, while `smooth` ($\mu$ = 100) found an interior θ = 0.70.
- **Options considered:** (a) report as is; (b) change `stress_short` itself; (c) keep `stress_short` unchanged and
  add `stress_short_wide`, identical except that the θ grid runs to 3.0.
- **Decision:** (c). The trigger is a validation-only rule (selected θ on the grid edge), but the check was added
  after test results were seen, so it is reported as a post-hoc sensitivity analysis next to the original.
- **Consequences / follow-ups:** the report must quote both runs. The same check was then applied to `final.json`
  and `stress_timbre.json` (`final_wide.json`, `stress_timbre_wide.json`): at σ = 0.20 the main baseline's validation
  curve also rises at 0.95 although its selected θ = 0.35 is inside the grid. Results: the main σ = 0.20 baseline rises
  0.545 → 0.586 (θ = 1.05); the penalty stays neutral (+0.0007 [−0.0035, +0.0048]); the post-hoc filters help
  (+0.013 mean, +0.021 median). σ = 0.05 / 0.10 and the timbre case are unchanged. The main configs keep the guide's
  grid; the wide runs are reported as sensitivity analyses.

### 2026-11-28 — Post-hoc smoothing control and stress cases

- **Control:** `posthoc_mean` (moving average, the linear analogue of the quadratic penalty) and `posthoc_median`
  (running median, edge-preserving), applied to baseline activations before thresholding, with the window
  (3, 5, 9, 15, 25 frames) selected jointly with θ on validation. Same selection rule as $\mu$. Code:
  `src/postprocess.py`, `evaluate_setting(window_grid=..., filter_kind=...)`.
- **Stress cases:** `stress_short` (notes 0.10–0.20 s) and `stress_timbre` (harmonic amplitudes $1/h^2$) change the
  validation and test clips only; the dictionary, protocol and methods are those of `final.json` (a test enforces
  this). Validation clips are stressed too, so every method is re-tuned for the condition it is tested on.
- **Not run:** notes that always coincide. With two pitches always together the alignment step has no
  single-pitch frames to build a reference from, so the protocol cannot run; the failure is in dictionary learning,
  before the penalty acts. Covered by the blind joint factorization instead.
- **Data generator:** `make_clip` gained `duration_range`, `harmonic_amps` and `coincide`. Defaults are bit-identical
  to the guide's generator (`test_default_reproduces_make_music_exactly`), and every random draw stays in the same
  order, so a stressed clip keeps its seed's phases and noise.

### 2026-11-28 — Joint $\mu$/$\theta$ selection on validation; ablation as a separate method

- **Context:** the modified method has a second setting, $\mu$, whose effective strength depends on the scale of
  $H$. That scale grows with noise (validation clip 100: mean $H$ ≈ 0.45 at σ = 0.05, ≈ 1.04 at σ = 0.20).
- **Options considered:** (a) one fixed $\mu$ for all noise levels; (b) $\mu$ normalized by mean($V$);
  (c) $\mu$ selected per noise level on validation clips, together with $\theta$.
- **Decision:** (c), on the grid 0.3, 1, 3, 10, 30, 100. For each $\mu$ the best $\theta$ is found; the $\mu$ with
  the highest validation mean $F_1$ wins, ties to the smallest. $\mu=0$ is not in the grid. It runs as its own method,
  `smooth_mu0`, through the modified code path.
- **Why:** (c) treats $\mu$ like $\theta$: a setting chosen from validation data per condition, so no scale constant has
  to be fixed by hand. Leaving out $\mu=0$ keeps "smooth" a test of the penalty rather than a choice between
  smoothing and none. The grid came from the convergence check on validation clip 100, before any test run.
- **Consequences / follow-ups:** the modified method searches 114 ($\mu$, $\theta$) pairs against the baseline's 19
  (see Section 3). `run_info.json["validation"]` stores the full ($\mu$, $\theta$) F1 table. Implemented in
  `src/protocol.py::evaluate_setting(mu_grid=...)`, `run_experiments.py` (`methods`, `paired.csv`) and
  `configs/final.json`.

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
- Fresh clone of `3e68fd9`: 203 passed, 1 skipped; smoke run and figures OK.
- Tagged `v0.1-baseline` (annotated) on `3e68fd9`.
- **Next:** start `modified.py`.

### 2026-11-28

- Implemented `src/modified.py` (smoothness-penalized H update, inference) and `tests/test_modified.py`.
- Protocol: `evaluate_setting(..., mu_grid=...)` selects $\mu$ and $\theta$ on validation; `run_experiments.py` reads
  a `methods` list and writes `paired.csv`; added `configs/final.json`; smoke config now runs both methods.
- First full run of `configs/final.json`: validation picks the smallest $\mu$ (0.3) at every noise level, and
  validation $F_1$ falls steadily as $\mu$ grows. See Section 11.
- Validation-only error breakdown (baseline, seeds 100–109): at σ = 0.05 every error lies within 3 frames of a note
  boundary; at σ = 0.20 almost every error is a false positive in silence (2972 of 3123 error frames), and
  predictions change *less* often than the labels (80 vs 120 changes). The off-note 95th percentile of the scaled
  activations is ≈ 1.0, which means a raised floor rather than frame-to-frame jitter. A smoothness penalty cannot lower
  a floor.
- Added the post-hoc control (`src/postprocess.py`), stress options and configs, the paired / validation / per-method
  example figures, and `docs/report_notes.md` (hypothesis, results, interpretation).
- Found that the standard threshold grid limits the baseline at σ = 0.20; added `*_wide` sensitivity runs (decision log).
- **Next:** review `docs/report_notes.md`; finalize the assistance statement; write the report; tag `v1.0-submission`.

## 10. Open questions

- [x] Is the simple split-gradient $H$ update monotone, or is the exact quadratic form needed? — monotone in every
  check so far (Section 4); the exact form is not needed.
- [x] Which noise levels give a useful spread of baseline $F_1$? — 0.05 / 0.10 / 0.20 (see decision log).
- [ ] The selected $\theta$ is not monotone in noise (0.40, 0.55, 0.35). Why does it drop at 0.20? Look at the
  validation $F_1$ curves in `run_info.json["validation"]`. — Likely lead (2026-11-28): at σ = 0.20 the off-note
  activations sit near 1.0 (the note templates explain the broadband noise), so no threshold separates notes from
  silence and the best $F_1$ comes from a low $\theta$ that keeps every note.

## 11. Results summary

*Fill after the final run. State the primary-metric result honestly, including a null or negative one.*

Baseline only so far (`configs/baseline.json`, 30 test clips per setting, mean and 95% interval):

| Noise std | $\theta$ | Frame $F_1$          | $D/(FT)$ | Mean \|onset lag\| (frames) | Onsets missed |
|-----------|----------|----------------------|----------|-----------------------------|---------------|
| 0.05      | 0.40     | 0.988 [0.985, 0.990] | 0.00270  | 0.12 [0.07, 0.16]           | 0             |
| 0.10      | 0.55     | 0.933 [0.925, 0.941] | 0.00579  | 0.46 [0.39, 0.54]           | 0             |
| 0.20      | 0.35     | 0.545 [0.536, 0.554] | 0.01233  | 2.69 [2.67, 2.71]           | 0             |

**Final comparison** (2026-11-28). Full tables, stress cases, the threshold-range sensitivity check and the
interpretation are in [`report_notes.md`](report_notes.md), Section 5. Summary:

- **Primary metric:** the penalty never improves $F_1$. $\Delta F_1$ = −0.0040 [−0.0049, −0.0032] at σ = 0.05,
  −0.0040 [−0.0059, −0.0021] at 0.10, −0.0000 [−0.0005, +0.0005] at 0.20 (30 paired clips each, `results/final/paired.csv`).
  Validation picks the smallest $\mu$ (0.3) everywhere.
- **Onset smearing:** +0.16 and +0.25 frames at σ = 0.05 and 0.10 (predicted harm confirmed).
- **Ablation:** `smooth_mu0` identical to the baseline.
- **Control:** the running median helps at σ = 0.10 (+0.0047 [+0.0034, +0.0060]); the moving average behaves like the
  penalty.
- **Stress:** short notes double the penalty's cost at σ = 0.05 / 0.10; changed timbre: no recovery. The short-note
  σ = 0.20 "win" (+0.11) is a threshold-grid artefact and reverses to −0.05 with a wider grid.
- **Runtime:** +1–3 ms per clip (3–8%); ≈ 1 ms differences are within timing noise.
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
    - T. Virtanen (2007), IEEE TASLP 15(3), 1066–1074 — temporal continuity (smoothness) penalty.
    - `scipy.ndimage.uniform_filter1d`, `scipy.ndimage.median_filter` — post-hoc smoothing control.
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
    - Modification milestone (2026-11-28): `src/modified.py` and `tests/test_modified.py`; the protocol and runner
      extensions (`mu_grid`, `window_grid`, `clip_kwargs`, `paired.csv`); `src/postprocess.py`; the stress options in
      `src/data.py`; `configs/final.json` and the stress configs; the paired, validation and per-method example
      figures in `make_figures.py`; the related tests; `docs/report_notes.md`
