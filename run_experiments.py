"""Run the baseline and modified protocols from a JSON configuration.

Usage, from the repository root::

    python run_experiments.py --config configs/smoke.json      # seconds
    python run_experiments.py --config configs/baseline.json   # baseline only
    python run_experiments.py --config configs/final.json      # baseline vs modified

The dictionary is learned once on development clips at the default noise level
(:func:`src.protocol.learn_dictionary`) and reused at every noise level in
``config["noise_levels"]`` and for every method. Per level and method, the
threshold (and, for the modified method, the penalty weight ``mu``) is selected
on validation clips (:func:`src.protocol.evaluate_setting`). Test labels enter
only at scoring.

Methods
-------
``config["methods"]`` is an optional list of ``{"label": str}`` entries, each
with an optional ``"mu_grid"``. An entry without ``"mu_grid"`` runs baseline
inference; an entry with one runs smoothed inference with ``mu`` selected from
that grid (``[0.0]`` is the ablation: modified code path, penalty off). Without
``"methods"`` the run is baseline only, labelled ``"baseline"``.

Outputs, written into ``config["output_dir"]`` (created if needed)
------------------------------------------------------------------
``cases.csv``        long format, one row per (test clip, metric) with columns
                     ``case_id, split, seed, noise_std, method, metric, value,
                     runtime_s``.
``summary.csv``      one row per (noise level, method, metric) with the mean
                     and 95% t-interval over clips (Eq. 16): ``mean, low, high,
                     sd, n``.
``paired.csv``       only if a method labelled ``"baseline"`` and at least one
                     other method ran: one row per (noise level, method,
                     metric) with the mean paired difference ``method -
                     baseline`` over test clips and its 95% t-interval:
                     ``noise_std, method, reference, metric, mean_diff, low,
                     high, sd, n``. ``runtime_s`` is included as a metric.
                     Pairs where either value is NaN are dropped.
``run_info.json``    full configuration, selected thresholds and ``mu``,
                     validation curves, best restart
                     seed, matched similarities, scales, setup time, software
                     versions, platform and BLAS thread settings.
``example_clip.npz`` one fixed validation clip: ``V``, ``Y``, ``H_scaled``,
                     ``theta``, ``times``, ``freq``, plus the aligned ``W``,
                     the reference templates ``R`` and the ``matched``
                     similarities, so ``make_figures.py`` needs no recomputation.
                     ``H_scaled`` and ``theta`` belong to the reference method
                     (``"baseline"`` if present, else the first method); every
                     other method adds ``H_scaled__<label>``,
                     ``theta__<label>`` and ``mu__<label>``.
"""
import os

# Pin BLAS threads for stable timings (guide 6.4). Must run before NumPy is imported.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
for _var in _THREAD_VARS:
    os.environ.setdefault(_var, "1")

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from src.alignment import reference_templates
from src.data import concat_clips, make_clip, split_seeds
from src.protocol import infer_method, evaluate_setting, learn_dictionary
from src.stats import mean_ci

CASE_COLUMNS = ["case_id", "split", "seed", "noise_std", "method", "metric", "value", "runtime_s"]
SUMMARY_COLUMNS = ["noise_std", "method", "metric", "mean", "low", "high", "sd", "n"]
PAIRED_COLUMNS = ["noise_std", "method", "reference", "metric", "mean_diff", "low", "high", "sd", "n"]
REFERENCE = "baseline"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the KL-NMF baseline and modified protocols.")
    parser.add_argument("--config", required=True, type=Path,
                        help="path to a JSON configuration, e.g. configs/smoke.json")
    args = parser.parse_args(argv)
    if not args.config.is_file():
        parser.error(f"config file not found: {args.config}")
    return args


def load_config(path):
    """Read a JSON configuration file into a dict."""
    with open(path) as f:
        return json.load(f)


def parse_methods(config):
    """Read ``config["methods"]`` into a list of ``{"label", "mu_grid"}`` dicts.

    ``mu_grid`` is ``None`` for baseline inference. Without ``"methods"`` the
    result is ``[{"label": "baseline", "mu_grid": None}]``.

    Raises
    ------
    ValueError
        If the list is empty, an entry has no label, or labels repeat.
    """
    entries = config.get("methods", [{"label": REFERENCE}])
    if not entries:
        raise ValueError("config['methods'] must not be empty.")
    methods = []
    for entry in entries:
        if "label" not in entry:
            raise ValueError(f"method entry without a label: {entry}")
        grid = entry.get("mu_grid")
        methods.append({"label": str(entry["label"]),
                        "mu_grid": None if grid is None else [float(m) for m in grid]})
    labels = [m["label"] for m in methods]
    if len(set(labels)) != len(labels):
        raise ValueError(f"method labels must be unique: {labels}")
    return methods


def to_jsonable(obj):
    """Recursively convert NumPy scalars/arrays and paths into JSON-friendly types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records):
    """Aggregate per-clip records into one row per (noise level, method, metric).

    Uses :func:`src.stats.mean_ci` (Student-t 95% interval over clips, Eq. 16).
    The clip is the unit of analysis, so ``n`` is the number of test clips.
    """
    groups = {}
    for r in records:
        groups.setdefault((r["noise_std"], r["method"], r["metric"]), []).append(r["value"])
    rows = []
    for (noise_std, method, metric), values in groups.items():
        ci = mean_ci(np.asarray(values, dtype=float))
        rows.append({
            "noise_std": noise_std,
            "method": method,
            "metric": metric,
            "mean": float(ci["mean"]),
            "low": float(ci["low"]),
            "high": float(ci["high"]),
            "sd": float(ci["sd"]),
            "n": int(ci["n"]),
        })
    return rows


def paired_differences(records, reference=REFERENCE):
    """Per-clip paired differences ``method - reference``, summarized per setting.

    Cases are paired by (noise level, test seed). Each metric and the runtime
    are compared; a pair is dropped if either value is NaN (e.g. a mean onset
    lag when every onset is missed), so ``n`` can be below the clip count.
    The interval is the Student-t 95% interval of :func:`src.stats.mean_ci`
    over the per-clip differences; the clip is the unit of analysis.

    Returns
    -------
    list of dict
        Rows with keys ``PAIRED_COLUMNS``; empty if ``reference`` is absent or
        is the only method.
    """
    values = {}
    for r in records:
        values[(r["noise_std"], r["method"], r["metric"], r["seed"])] = r["value"]
        values[(r["noise_std"], r["method"], "runtime_s", r["seed"])] = r["runtime_s"]

    methods = sorted({r["method"] for r in records} - {reference})
    if reference not in {r["method"] for r in records}:
        return []
    keys = sorted({(k[0], k[2]) for k in values})  # (noise_std, metric)
    rows = []
    for method in methods:
        for noise_std, metric in keys:
            seeds = sorted({k[3] for k in values if k[:3] == (noise_std, method, metric)})
            diffs = [values[(noise_std, method, metric, s)] - values[(noise_std, reference, metric, s)]
                     for s in seeds if (noise_std, reference, metric, s) in values]
            diffs = np.asarray([d for d in diffs if np.isfinite(d)], dtype=float)
            if diffs.size == 0:
                ci = {"mean": np.nan, "low": np.nan, "high": np.nan, "sd": np.nan, "n": 0}
            else:
                ci = mean_ci(diffs)
            rows.append({
                "noise_std": noise_std,
                "method": method,
                "reference": reference,
                "metric": metric,
                "mean_diff": float(ci["mean"]),
                "low": float(ci["low"]),
                "high": float(ci["high"]),
                "sd": float(ci["sd"]),
                "n": int(ci["n"]),
            })
    return rows


def dev_reference_templates(config):
    """Reference templates r_k (Eq. 10) from development clips only, for the figures."""
    dev_seeds = split_seeds(config["seeds"])["dev"]
    dev = concat_clips([make_clip(s, noise_std=config["dev_noise_std"]) for s in dev_seeds])
    return reference_templates(dev["V"], dev["Y"])["R"]


def save_example_clip(path, dictionary, R, config, results, methods):
    """Save one fixed validation clip with everything needed to redraw its figure.

    The clip is the first validation seed at the first configured noise level,
    inferred exactly as in :func:`src.protocol.evaluate_setting` (clip seed,
    ``n_iter_infer`` iterations, frozen ``W``, the selected ``mu``) and scaled
    by ``c_k``. ``results[(noise_std, label)]`` holds each method's selected
    ``"threshold"`` and ``"mu"``.
    """
    seed = split_seeds(config["seeds"])["val"][0]
    noise_std = float(config["noise_levels"][0])
    clip = make_clip(seed, noise_std=noise_std)
    W = np.asarray(dictionary["W"])
    scales = np.asarray(dictionary["scales"])[:, None]

    labels = [m["label"] for m in methods]
    ref = REFERENCE if REFERENCE in labels else labels[0]
    arrays = {}
    for label in labels:
        res = results[(noise_std, label)]
        H_scaled = infer_method(clip["V"], W, config, seed, res["mu"]) / scales
        if label == ref:
            arrays.update(H_scaled=H_scaled, theta=res["threshold"])
        else:
            arrays[f"H_scaled__{label}"] = H_scaled
            arrays[f"theta__{label}"] = res["threshold"]
            arrays[f"mu__{label}"] = np.nan if res["mu"] is None else res["mu"]

    np.savez_compressed(
        path,
        V=clip["V"], Y=clip["Y"], times=clip["times"], freq=clip["freq"],
        W=W, R=R, matched=np.asarray(dictionary["match"]["matched"], dtype=float),
        seed=seed, noise_std=noise_std, reference_method=ref, **arrays,
    )
    return {"seed": seed, "noise_std": noise_std, "reference_method": ref}


def environment_info():
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "threads": {var: os.environ.get(var) for var in _THREAD_VARS},
    }


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{config.get('name', args.config.stem)}] learning dictionary on development clips ...",
          flush=True)
    dictionary = learn_dictionary(config)
    print(f"  best seed {dictionary['best_seed']}, "
          f"matched similarities {np.round(dictionary['match']['matched'], 4)}, "
          f"setup {dictionary['setup_seconds']:.1f} s", flush=True)

    methods = parse_methods(config)
    records, results = [], {}
    for noise_std in config["noise_levels"]:
        noise_std = float(noise_std)
        for m in methods:
            result = evaluate_setting(dictionary, noise_std, config, method=m["label"],
                                      mu_grid=m["mu_grid"])
            records.extend(result["records"])
            results[(noise_std, m["label"])] = result
            f1 = [r["value"] for r in result["records"] if r["metric"] == "f1"]
            mu = "" if result["mu"] is None else f", mu {result['mu']:g}"
            print(f"  noise {noise_std:g} [{m['label']}]: theta {result['threshold']:.2f}{mu}, "
                  f"test F1 mean {np.mean(f1):.3f}", flush=True)

    write_csv(out_dir / "cases.csv", records, CASE_COLUMNS)
    write_csv(out_dir / "summary.csv", summarize(records), SUMMARY_COLUMNS)
    paired = paired_differences(records)
    if paired:
        write_csv(out_dir / "paired.csv", paired, PAIRED_COLUMNS)

    R = dev_reference_templates(config)
    example = save_example_clip(out_dir / "example_clip.npz", dictionary, R, config, results, methods)

    run_info = {
        "config_path": args.config,
        "config": config,
        "methods": methods,
        "thresholds": [{"noise_std": k[0], "method": k[1], "threshold": v["threshold"], "mu": v["mu"]}
                       for k, v in results.items()],
        "validation": [{"noise_std": k[0], "method": k[1], "grid": v["grid"],
                        "mean_f1": v["val_mean_f1"], "mu_grid": v["mu_grid"],
                        "mean_f1_by_mu": v["val_mean_f1_by_mu"]}
                       for k, v in results.items()],
        "best_seed": dictionary["best_seed"],
        "final_losses": dictionary["final_losses"],
        "match": dictionary["match"],
        "scales": dictionary["scales"],
        "setup_seconds": dictionary["setup_seconds"],
        "example_clip": example,
        "environment": environment_info(),
    }
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(to_jsonable(run_info), f, indent=2)

    print(f"  wrote results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
