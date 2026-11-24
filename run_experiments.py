"""Run the baseline protocol from a JSON configuration (Project Guide A1, Section 6.5).

Usage, from the repository root::

    python run_experiments.py --config configs/smoke.json      # seconds
    python run_experiments.py --config configs/baseline.json   # about 1-2 min

The dictionary is learned once on development clips at the default noise level
(:func:`src.protocol.learn_dictionary`) and reused at every noise level in
``config["noise_levels"]``; the threshold is re-selected per level on validation
clips (:func:`src.protocol.evaluate_setting`). Test labels enter only at scoring.

Outputs, written into ``config["output_dir"]`` (created if needed)
------------------------------------------------------------------
``cases.csv``        long format, one row per (test clip, metric) with columns
                     ``case_id, split, seed, noise_std, method, metric, value,
                     runtime_s``.
``summary.csv``      one row per (noise level, method, metric) with the mean
                     and 95% t-interval over clips (Eq. 16): ``mean, low, high,
                     sd, n``.
``run_info.json``    full configuration, selected thresholds, best restart
                     seed, matched similarities, scales, setup time, software
                     versions, platform and BLAS thread settings.
``example_clip.npz`` one fixed validation clip: ``V``, ``Y``, ``H_scaled``,
                     ``theta``, ``times``, ``freq``, plus the aligned ``W``,
                     the reference templates ``R`` and the ``matched``
                     similarities, so ``make_figures.py`` needs no recomputation.
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
from src.baseline import infer_activations
from src.data import concat_clips, make_clip, split_seeds
from src.protocol import evaluate_setting, learn_dictionary
from src.stats import mean_ci

CASE_COLUMNS = ["case_id", "split", "seed", "noise_std", "method", "metric", "value", "runtime_s"]
SUMMARY_COLUMNS = ["noise_std", "method", "metric", "mean", "low", "high", "sd", "n"]
METHOD = "baseline"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the KL-NMF baseline protocol.")
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


def dev_reference_templates(config):
    """Reference templates r_k (Eq. 10) from development clips only, for the figures."""
    dev_seeds = split_seeds(config["seeds"])["dev"]
    dev = concat_clips([make_clip(s, noise_std=config["dev_noise_std"]) for s in dev_seeds])
    return reference_templates(dev["V"], dev["Y"])["R"]


def save_example_clip(path, dictionary, R, config, thresholds):
    """Save one fixed validation clip with everything needed to redraw its figure.

    The clip is the first validation seed at the first configured noise level,
    inferred exactly as in :func:`src.protocol.evaluate_setting` (clip seed,
    ``n_iter_infer`` iterations, frozen ``W``) and scaled by ``c_k``.
    """
    seed = split_seeds(config["seeds"])["val"][0]
    noise_std = float(config["noise_levels"][0])
    clip = make_clip(seed, noise_std=noise_std)
    W = np.asarray(dictionary["W"])
    H = infer_activations(clip["V"], W, n_iter=config["n_iter_infer"], seed=seed,
                          eps=config["eps"], track_loss=False)["H"]
    H_scaled = H / np.asarray(dictionary["scales"])[:, None]
    np.savez_compressed(
        path,
        V=clip["V"], Y=clip["Y"], H_scaled=H_scaled, theta=thresholds[noise_std],
        times=clip["times"], freq=clip["freq"],
        W=W, R=R, matched=np.asarray(dictionary["match"]["matched"], dtype=float),
        seed=seed, noise_std=noise_std,
    )
    return {"seed": seed, "noise_std": noise_std}


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

    records, thresholds, validation = [], {}, {}
    for noise_std in config["noise_levels"]:
        noise_std = float(noise_std)
        result = evaluate_setting(dictionary, noise_std, config, method=METHOD)
        records.extend(result["records"])
        thresholds[noise_std] = float(result["threshold"])
        validation[noise_std] = {"grid": result["grid"], "mean_f1": result["val_mean_f1"]}
        f1 = [r["value"] for r in result["records"] if r["metric"] == "f1"]
        print(f"  noise {noise_std:g}: theta {thresholds[noise_std]:.2f}, "
              f"test F1 mean {np.mean(f1):.3f}", flush=True)

    write_csv(out_dir / "cases.csv", records, CASE_COLUMNS)
    write_csv(out_dir / "summary.csv", summarize(records), SUMMARY_COLUMNS)

    R = dev_reference_templates(config)
    example = save_example_clip(out_dir / "example_clip.npz", dictionary, R, config, thresholds)

    run_info = {
        "config_path": args.config,
        "config": config,
        "method": METHOD,
        "thresholds": [{"noise_std": k, "threshold": v} for k, v in thresholds.items()],
        "validation": [{"noise_std": k, **v} for k, v in validation.items()],
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
