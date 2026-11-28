import time

import numpy as np

from src.data import split_seeds, make_clip, concat_clips
from src.baseline import fit_nmf_restarts, infer_activations
from src.modified import infer_activations_smooth
from src.alignment import reference_templates, match_components, activation_scales
from src.metrics import frame_f1, kl_per_entry, onset_lags, select_threshold


def learn_dictionary(config):
    """Learn the aligned, frozen dictionary from development clips.

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


def infer_method(V, W, config, seed, mu=None):
    """Infer activations with frozen ``W``: baseline if ``mu is None``, else smoothed.

    Both methods use the same initialization for the same ``seed``, the same
    iteration budget ``n_iter_infer`` and the same ``eps``.
    """
    if mu is None:
        return infer_activations(V, W, n_iter=config["n_iter_infer"], seed=seed,
                                 eps=config["eps"], track_loss=False)["H"]
    return infer_activations_smooth(V, W, mu, n_iter=config["n_iter_infer"], seed=seed,
                                    eps=config["eps"], track_loss=False)["H"]


def _timed_inference(V, W, config, seed, mu=None):
    """Infer activations (see :func:`infer_method`), timed.

    Runs ``warmup`` untimed calls, then ``repeats`` timed calls with
    :func:`time.perf_counter`, and returns the activations and the median time.
    """
    timing = config.get("timing", {})
    warmup, repeats = int(timing.get("warmup", 1)), int(timing.get("repeats", 5))

    for _ in range(warmup):
        infer_method(V, W, config, seed, mu)
    times = []
    for _ in range(max(repeats, 1)):
        t0 = time.perf_counter()
        H = infer_method(V, W, config, seed, mu)
        times.append(time.perf_counter() - t0)
    return H, float(np.median(times))


def _check_mu_grid(mu_grid):
    """Return ``mu_grid`` as a sorted, de-duplicated float array, or raise ``ValueError``."""
    grid = np.unique(np.asarray(mu_grid, dtype=float))  # sorted ascending
    if grid.size == 0:
        raise ValueError("mu_grid must not be empty.")
    if not np.all(np.isfinite(grid)) or np.any(grid < 0):
        raise ValueError("mu_grid values must be finite and non-negative.")
    return grid


def evaluate_setting(dictionary, noise_std, config, method="baseline", mu_grid=None):
    """Evaluate a frozen dictionary at one noise level.

    With ``mu_grid=None`` this is the baseline: activations come from
    :func:`src.baseline.infer_activations` and only the threshold is selected.
    With a ``mu_grid`` it is the modified method: activations come from
    :func:`src.modified.infer_activations_smooth`, and the penalty weight
    ``mu`` and the threshold are selected together. For each ``mu`` the best
    threshold is found, and the ``mu`` with the highest validation mean F1 is
    kept (ties go to the smallest ``mu``, i.e. the least smoothing).

    All settings are selected on validation clips only; test labels are used
    only for the final scoring. Inference on each clip uses the clip seed, so
    baseline and modified inference start from identical activations.

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
    mu_grid : array_like of float, optional
        Candidate penalty weights (finite, >= 0) for the modified method.
        ``None`` (default) runs the baseline. ``[0.0]`` runs the modified code
        path with the penalty switched off (the ablation).

    Returns
    -------
    dict
        ``"records"`` : list of dict -- one per (test clip, metric) with keys
        ``case_id``, ``split``, ``seed``, ``noise_std``, ``method``,
        ``metric``, ``value``, ``runtime_s``. Metrics are ``f1``,
        ``kl_per_entry``, ``onset_mean_abs_lag_frames`` (NaN if every onset
        is missed) and ``onsets_missed``.
        ``"threshold"`` : float -- threshold selected on validation clips.
        ``"val_mean_f1"`` : ndarray -- mean validation F1 for each threshold
        in ``grid`` (at the selected ``mu`` for the modified method).
        ``"grid"`` : ndarray -- the threshold grid.
        ``"mu"`` : float or None -- selected penalty weight (None for the
        baseline).
        ``"mu_grid"`` : ndarray or None -- sorted candidate weights.
        ``"val_mean_f1_by_mu"`` : ndarray or None -- shape
        ``(len(mu_grid), len(grid))``, mean validation F1 for every
        (``mu``, threshold) pair.

    Raises
    ------
    ValueError
        If ``mu_grid`` is empty or has negative or non-finite values.
    """
    W = np.asarray(dictionary["W"])
    scales = np.asarray(dictionary["scales"])[:, None]
    seeds = split_seeds(config["seeds"])
    eps = config["eps"]

    theta_grid = _threshold_grid(config["threshold_grid"])
    mus = [None] if mu_grid is None else list(_check_mu_grid(mu_grid))

    # 5. Validation clips: infer with frozen W, scale, and select mu (modified only)
    #    and the threshold. Clips are generated once and reused for every mu.
    val_clips = [(int(s), make_clip(int(s), noise_std=noise_std)) for s in seeds["val"]]
    Y_val = [clip["Y"].astype(bool) for _, clip in val_clips]
    selections = []
    for mu in mus:
        H_val = [infer_method(clip["V"], W, config, s, mu) / scales for s, clip in val_clips]
        selections.append(select_threshold(H_val, Y_val, theta_grid))
    best_f1 = [float(np.max(sel["mean_f1"])) for sel in selections]
    best = int(np.argmax(best_f1))  # first maximum -> smallest mu on ties
    mu = mus[best]
    selection = selections[best]
    theta = float(selection["threshold"])

    # 6. Test clips: infer (timed), scale, threshold and score.
    records = []
    for s in seeds["test"]:
        s = int(s)
        clip = make_clip(s, noise_std=noise_std)
        V, Y = clip["V"], clip["Y"].astype(bool)
        H, runtime = _timed_inference(V, W, config, s, mu)
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
        "mu": None if mu is None else float(mu),
        "mu_grid": None if mu_grid is None else np.asarray(mus, dtype=float),
        "val_mean_f1_by_mu": None if mu_grid is None else np.vstack([sel["mean_f1"] for sel in selections]),
    }
