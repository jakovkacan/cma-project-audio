# Test suite: Track A baseline (Project Guide A1)

Local checks for every function in the baseline guide. Run them from the **repository root**.

```bash
python -m pip install pytest==9.1.1        # or add it to requirements-dev.txt
python -m pytest                            # everything, about 1 minute
python -m pytest -m "not slow"              # quick loop, about 15 s
python -m pytest tests/test_baseline.py     # one module
python -m pytest -rs                        # also list skipped tests and why
python -m pytest -x --lf                    # stop at the first failure; rerun last failures
```

## How it behaves while you build

- A **module that does not exist yet** (for example `src/stats.py`) makes its tests *skip*. You can implement the guide section by section and keep the suite green.
- **Any other import problem** (a syntax error, a missing name, a bad import inside your module) is an *error*. It is never skipped.
- `test_repository.py` **fails** for every required file that is still missing, so skipped modules can't slip through before submission.

## What each file checks

| File | Guide section | Covers |
|---|---|---|
| `test_data.py` | 2 | `make_music`, `make_clip` (keys, shapes, axes, noise pairing, validation), `split_seeds`, `concat_clips` |
| `test_baseline.py` | 3 | `kl_divergence`, `init_factors` (exact spec), both updates (hand example, matrix form, fixed point, monotonicity), `normalize_templates`, `fit_nmf` (equivalence to the spec loop, rank-one closed form, exact low-rank recovery, zeros, validation), `fit_nmf_restarts`, `infer_activations` |
| `test_alignment.py` | 4 | `reference_templates`, `cosine_similarity`, `match_components` (cyclic permutation, optimality, extra non-note component), `activation_scales` |
| `test_metrics.py` | 5 | `frame_f1`, `kl_per_entry`, `onset_lags` (late, early, $t=0$, missed, window edges), `select_threshold` (per-clip mean, tie rule) |
| `test_stats_timing.py` | 6.3–6.4 | `mean_ci`, `time_call` |
| `test_protocol.py` | 6.1 | `learn_dictionary`, `evaluate_setting` on a tiny config, plus **leakage checks**: only development seeds at `dev_noise_std` for the dictionary; the threshold does not change when only the test seeds change |
| `test_runner.py` | 6.5, 7 | `run_experiments.py` and `make_figures.py` in a subprocess, writing to a temporary folder: CSV columns, row counts, summary consistency, `run_info.json`, `example_clip.npz`, three PNGs |
| `test_repository.py` | Notes, 6.2 | Required files, pinned requirements, no scikit-learn or matplotlib inside `src/`, config keys and values |
| `test_reference_values.py` | 3.5, 4, 6.1 | The numbers quoted in the guide (marked `slow`) |

## Conventions the tests assume

Where the guide leaves a detail open, the tests accept either reasonable choice, or pick one and say so:

- `evaluate_setting` may return `{"records": [...], "threshold": t}` (preferred) or `(records, threshold)`. Each record is a dict with the `cases.csv` columns, one per (test clip, metric).
- The leakage tests intercept `make_clip` (both `src.data.make_clip` and a `make_clip` imported into `src.protocol`), so generate clips through `make_clip`.
- `example_clip.npz` must contain at least `V` and `Y` and five arrays in total. `make_figures.py` must also be able to draw `fig_templates.png`, so save the aligned $W$ and the references $R$ somewhere in the results folder (for example in the same `.npz`).
- `best_seed` in the slow test may be 1, 2 or 3: those restarts agree to about five significant figures.
- `tests/README.md` (this file) is documentation only; pytest ignores it.

## Notes

- The suite was checked against an independent reference implementation of the guide: all tests pass. It was also checked with 19 deliberately planted bugs, and every one was caught. Examples: drawing $H$ before $W$, reusing a stale $Q$, renormalizing during frozen inference, the wrong tie rule, pooling frames for $F_1$, choosing $\theta$ on test clips, and `ddof=0` in the interval.
- Passing these tests shows the code meets the stated contracts. It does not show that the model is valid: magnitudes only approximately add, and components need not be notes. Keep that distinction in the report.
- These tests are your verification evidence, so mention them in the README and the report, and include them in your assistance declaration.
