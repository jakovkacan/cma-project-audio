import time

import numpy as np

from src.data import split_seeds, make_clip, concat_clips
from src.baseline import fit_nmf_restarts, infer_activations
from src.alignment import reference_templates, match_components, activation_scales
from src.metrics import frame_f1, kl_per_entry, onset_lags, select_threshold


def learn_dictionary(config):
    """Learn the aligned, frozen dictionary from development clips (guide 6.1, steps 1-4).

    Only development seeds are used, so validation and test data never
    influence the templates or the scales.

    Parameters
    ----------
    config : dict
        Experiment configuration. Uses the keys ``"seeds"`` (``"dev"`` interval),
        ``"dev_noise_std"``, ``"n_components"``, ``"restart_seeds"``,
        ``"n_iter_dict"``, ``"n_iter_infer"``, ``"eps"`` and ``"scale_percentile"``.

    Returns
    -------
    dict
        ``"W"`` : ndarray, shape (F, 3) -- templates in pitch order, columns
        summing to one (unassigned components are discarded).
        ``"scales"`` : ndarray, shape (3,) -- per-pitch activation scales.
        ``"match"`` : dict -- output of :func:`match_components`.
        ``"best_seed"`` : int -- restart seed with the lowest final loss.
        ``"final_losses"`` : ndarray -- final loss of each restart.
        ``"setup_seconds"`` : float -- wall time of the whole procedure.
    """
    start = time.perf_counter()
    eps = config["eps"]

    # 1. Generate development clips at the development noise level and concatenate them.
    dev_seeds = split_seeds(config["seeds"])["dev"]
    dev_clips = [make_clip(int(s), noise_std=config["dev_noise_std"]) for s in dev_seeds]
    dev = concat_clips(dev_clips)
    V, Y = dev["V"], dev["Y"]

    # 2. Fit with restarts; keep the lowest final loss.
    fit = fit_nmf_restarts(V, config["n_components"], config["restart_seeds"],
                           n_iter=config["n_iter_dict"], eps=eps)

    # 3. Build references, match and reorder W.
    R = reference_templates(V, Y)["R"]
    match = match_components(fit["W"], R)
    W_ord = fit["W"][:, match["perm"]]

    # 4. Infer development activations with the frozen, aligned W (seed equal to the
    #    clip seed), concatenate them and compute the per-pitch scales.
    H_dev = np.concatenate([
        infer_activations(clip["V"], W_ord, n_iter=config["n_iter_infer"],
                          seed=int(s), eps=eps, track_loss=False)["H"]
        for s, clip in zip(dev_seeds, dev_clips)
    ], axis=1)
    scales = activation_scales(H_dev, Y, q=config["scale_percentile"])

    return {
        "W": W_ord,
        "scales": scales,
        "match": match,
        "best_seed": fit["best_seed"],
        "final_losses": fit["final_losses"],
        "setup_seconds": time.perf_counter() - start,
    }


def _threshold_grid(spec):
    """Expand a ``{"start", "stop", "step"}`` spec into an inclusive grid."""
    grid = np.arange(spec["start"], spec["stop"] + spec["step"] / 2, spec["step"])
    return np.round(grid, 10)


def _timed_inference(V, W, config, seed):
    """Infer activations, timed as in guide 6.4 (``track_loss=False``).

    Runs ``warmup`` untimed calls, then ``repeats`` timed calls with
    :func:`time.perf_counter`, and returns the activations and the median time.
    """
    timing = config.get("timing", {})
    warmup, repeats = int(timing.get("warmup", 1)), int(timing.get("repeats", 5))

    def run():
        return infer_activations(V, W, n_iter=config["n_iter_infer"], seed=seed,
                                 eps=config["eps"], track_loss=False)["H"]

    for _ in range(warmup):
        run()
    times = []
    for _ in range(max(repeats, 1)):
        t0 = time.perf_counter()
        H = run()
        times.append(time.perf_counter() - t0)
    return H, float(np.median(times))


def evaluate_setting(dictionary, noise_std, config, method="baseline"):
    """Evaluate a frozen dictionary at one noise level (guide 6.1, steps 5-6).

    The threshold is selected on validation clips only; test labels are used
    only for the final scoring. Inference on each clip uses the clip seed.

    Parameters
    ----------
    dictionary : dict
        Output of :func:`learn_dictionary`; ``"W"`` and ``"scales"`` are used
        and not modified.
    noise_std : float
        Noise level used to generate the validation and test clips.
    config : dict
        Experiment configuration. Uses ``"seeds"`` (``"val"``, ``"test"``),
        ``"n_iter_infer"``, ``"eps"``, ``"threshold_grid"``, ``"onset_max_lag"``
        and ``"timing"``.
    method : str, optional
        Method label written into each record.

    Returns
    -------
    dict
        ``"records"`` : list of dict -- one per (test clip, metric) with keys
        ``case_id``, ``split``, ``seed``, ``noise_std``, ``method``,
        ``metric``, ``value``, ``runtime_s``. Metrics are ``f1``,
        ``kl_per_entry``, ``onset_mean_abs_lag_frames`` (NaN if every onset
        is missed) and ``onsets_missed``.
        ``"threshold"`` : float -- threshold selected on validation clips.
        ``"val_mean_f1"`` : ndarray -- mean validation F1 for each grid value.
        ``"grid"`` : ndarray -- the threshold grid.
    """
    W = np.asarray(dictionary["W"])
    scales = np.asarray(dictionary["scales"])[:, None]
    seeds = split_seeds(config["seeds"])
    eps = config["eps"]

    # 5. Validation clips: infer with frozen W, scale, and select the threshold.
    H_val, Y_val = [], []
    for s in seeds["val"]:
        clip = make_clip(int(s), noise_std=noise_std)
        H = infer_activations(clip["V"], W, n_iter=config["n_iter_infer"], seed=int(s),
                              eps=eps, track_loss=False)["H"]
        H_val.append(H / scales)
        Y_val.append(clip["Y"].astype(bool))
    selection = select_threshold(H_val, Y_val, _threshold_grid(config["threshold_grid"]))
    theta = float(selection["threshold"])

    # 6. Test clips: infer (timed), scale, threshold and score.
    records = []
    for s in seeds["test"]:
        s = int(s)
        clip = make_clip(s, noise_std=noise_std)
        V, Y = clip["V"], clip["Y"].astype(bool)
        H, runtime = _timed_inference(V, W, config, s)
        pred = (H / scales) > theta

        onsets = onset_lags(pred, Y, max_lag=config["onset_max_lag"])
        lags = onsets["lags"]
        values = {
            "f1": frame_f1(pred, Y)["f1"],
            "kl_per_entry": kl_per_entry(V, W, H, eps),
            "onset_mean_abs_lag_frames": float(np.mean(np.abs(lags))) if lags.size else float("nan"),
            "onsets_missed": onsets["missed"],
        }
        case_id = f"{method}_test_{s}_noise{noise_std:g}"
        for metric, value in values.items():
            records.append({
                "case_id": case_id,
                "split": "test",
                "seed": s,
                "noise_std": float(noise_std),
                "method": method,
                "metric": metric,
                "value": float(value),
                "runtime_s": runtime,
            })

    return {
        "records": records,
        "threshold": theta,
        "val_mean_f1": selection["mean_f1"],
        "grid": selection["grid"],
    }
