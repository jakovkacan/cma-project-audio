"""Figures from saved results: baseline, modification, control and stress runs

Reads only files saved by ``run_experiments.py`` and writes PNGs into
``<results>/figures/``. Nothing is recomputed: no factorization, no inference.

Usage, from the repository root::

    python make_figures.py --results results/baseline

Inputs expected in ``<results>/``
---------------------------------
``example_clip.npz`` -- one fixed validation clip:
    ``"V"``        magnitude spectrogram, shape (F, T), linear units;
    ``"Y"``        true pitch activity, shape (3, T), bool;
    ``"H_scaled"`` scaled activations ``H / c_k`` (Eq. 12), shape (3, T);
    ``"theta"``    selected threshold for that clip's noise level, scalar;
    ``"times"``    frame-centre times in s, shape (T,);
    ``"freq"``     bin frequencies in Hz, shape (F,);
    ``"W"``        aligned templates, shape (F, 3), columns sum to one;
    ``"R"``        reference templates ``r_k`` (Eq. 10), shape (F, 3);
    ``"matched"``  matched cosine similarities ``S_pi(k),k`` (Eq. 11), shape (3,).
    ``W``, ``R`` and ``matched`` may instead live in ``templates.npz``;
    ``matched`` may instead be ``run_info.json["match"]["matched"]``.
``cases.csv``   -- long format, one row per (case, metric); rows with
    ``metric == "f1"`` give the per-clip points.
``summary.csv`` -- one row per (noise level, method, metric) with
    ``mean, low, high`` (the 95% t-interval of Eq. 16).

Outputs in ``<results>/figures/``
---------------------------------
``fig_example.png``     3x1 panels on a shared time axis: log-magnitude
                        spectrogram ``20 log10(V + 1e-8)`` up to 2 kHz, true
                        activity ``Y``, and scaled activations with ``theta``.
``fig_templates.png``   aligned ``W`` against ``r_k`` up to 2 kHz, one panel
                        per pitch, matched similarity in each legend.
``fig_f1_by_noise.png`` per-clip test F1 (jittered) with mean and 95% interval,
                        one group per noise level, one colour and marker per method.

Written only when the run compared methods
------------------------------------------
``fig_paired.png``      needs ``paired.csv``: per-clip differences ``method -
                        baseline`` (jittered) with the mean and 95% interval,
                        for frame F1 and for mean |onset lag|, per noise level.
``fig_validation.png``  needs ``run_info.json["validation"]`` with a ``mu_grid``
                        or ``window_grid``: best validation F1 against the
                        tuned setting, one line per noise level, the baseline's
                        best validation F1 as a dashed line.
``fig_example_methods.png`` needs ``H_scaled__<label>`` in ``example_clip.npz``:
                        the example clip's scaled activations for every method
                        whose activations differ from the reference method's.

Method styles are fixed by label (``METHOD_STYLES``) and never cycled, so a
method keeps its colour in every figure and every run.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")  # file output only; never opens a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PITCH_NAMES = ["A3", "C♯4", "E4"]
PITCH_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed order: blue, orange, aqua
NEUTRAL = "#5f5e5a"
F_MAX_HZ = 2000.0
DB_FLOOR = 1e-8
DPI = 150
JITTER_SEED = 0
JITTER_HALF_WIDTH = 0.08

# Fixed per-label styles: (colour, marker, legend name). Categorical slots 1-4 of the
# dataviz reference palette (validated for colour-vision deficiency); the baseline is
# the neutral reference. Markers give a second, colour-free encoding.
METHOD_STYLES = {
    "baseline": (NEUTRAL, "o", "baseline"),
    "smooth": ("#257de0", "s", "smoothness penalty"),
    "posthoc_mean": ("#f47230", "^", "post-hoc moving average"),
    "posthoc_median": ("#00b894", "D", "post-hoc running median"),
    "smooth_mu0": ("#f2a600", "v", "ablation ($\\mu = 0$)"),
}
UNKNOWN_STYLE = ("#8a8984", "x", None)
# Ordinal noise levels: one blue ramp, light to dark (steps 250 / 450 / 650).
NOISE_RAMP = ["#84b9f3", "#257de0", "#0f4a96"]


def method_style(label):
    """Return ``(colour, marker, legend name)`` for a method label."""
    colour, marker, name = METHOD_STYLES.get(label, UNKNOWN_STYLE)
    return colour, marker, name or label


def method_order(labels):
    """Baseline first, then the known methods in ``METHOD_STYLES`` order, then the rest."""
    known = list(METHOD_STYLES)
    return sorted(labels, key=lambda m: (known.index(m) if m in known else len(known), m))


def noise_colour(i, n):
    """Colour for the ``i``-th of ``n`` ordered noise levels, from the blue ramp."""
    if n <= 1:
        return NOISE_RAMP[1]
    idx = np.linspace(0, len(NOISE_RAMP) - 1, n)
    return NOISE_RAMP[int(round(idx[i]))]


def _require(path):
    """Return ``path`` if it exists, otherwise exit with a clear message."""
    if not path.is_file():
        raise SystemExit(f"make_figures.py: missing {path} (run run_experiments.py first)")
    return path


def _save(fig, path):
    """Save as the guide prescribes and release the figure."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _style_axes(ax):
    """Recessive grid and spines so the data marks carry the figure."""
    ax.grid(True, color="#d9d8d3", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load_example(results):
    """Load the example clip and the aligned/reference templates.

    Returns a dict with keys ``"V"``, ``"Y"``, ``"H_scaled"``, ``"theta"``,
    ``"times"``, ``"freq"``, ``"W"``, ``"R"`` and ``"matched"`` (``None`` if
    the similarities were not saved anywhere).
    """
    with np.load(_require(results / "example_clip.npz")) as z:
        data = {name: z[name] for name in z.files}

    templates = results / "templates.npz"
    if templates.is_file():
        with np.load(templates) as z:
            for name in ("W", "R", "matched"):
                if name in z.files and name not in data:
                    data[name] = z[name]

    if "matched" not in data:
        info_path = results / "run_info.json"
        if info_path.is_file():
            match = json.loads(info_path.read_text()).get("match", {})
            if isinstance(match, dict) and "matched" in match:
                data["matched"] = np.asarray(match["matched"], dtype=float)

    required = ("V", "Y", "H_scaled", "theta", "times", "freq", "W", "R")
    missing = [name for name in required if name not in data]
    if missing:
        raise SystemExit(f"make_figures.py: example_clip.npz lacks {missing}")

    data["theta"] = float(np.asarray(data["theta"]))
    data.setdefault("matched", None)
    return data


def fig_example(ex, path):
    """Spectrogram, true activity and scaled activations on one time axis."""
    times, freq = ex["times"], ex["freq"]
    V, Y, H = ex["V"], np.asarray(ex["Y"], dtype=bool), ex["H_scaled"]
    theta = ex["theta"]
    band = freq <= F_MAX_HZ

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 0.7, 1.0]})

    # (a) log-magnitude spectrogram
    ax = axes[0]
    log_mag = 20.0 * np.log10(V[band] + DB_FLOOR)
    mesh = ax.pcolormesh(times, freq[band], log_mag, shading="nearest",
                         cmap="magma", rasterized=True)
    cax = ax.inset_axes([1.01, 0.0, 0.015, 1.0])  # keeps the x-extent aligned with panels below
    fig.colorbar(mesh, cax=cax, label="magnitude (dB)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("(a) Log-magnitude spectrogram $20\\log_{10}(V + 10^{-8})$, 0–2 kHz")

    # (b) true activity as a piano roll
    ax = axes[1]
    for k in range(Y.shape[0]):
        ax.fill_between(times, k - 0.35, k + 0.35, where=Y[k], step="mid",
                        color=PITCH_COLORS[k], linewidth=0)
    ax.set_yticks(range(Y.shape[0]), PITCH_NAMES[:Y.shape[0]])
    ax.set_ylim(-0.6, Y.shape[0] - 0.4)
    ax.set_ylabel("pitch")
    ax.set_title("(b) True activity $Y$")
    _style_axes(ax)
    ax.grid(False, axis="y")

    # (c) scaled activations with the selected threshold
    ax = axes[2]
    for k in range(H.shape[0]):
        ax.plot(times, H[k], color=PITCH_COLORS[k], linewidth=1.5, label=PITCH_NAMES[k])
    ax.axhline(theta, color=NEUTRAL, linestyle="--", linewidth=1.2,
               label=f"$\\theta$ = {theta:.2f}")
    ax.set_ylabel("scaled activation $\\tilde H$ (unitless)")
    ax.set_xlabel("time (s)")
    ax.set_title("(c) Scaled activations $\\tilde H_{kt} = H_{kt} / c_k$")
    ax.legend(loc="upper right", ncol=4, frameon=False)
    ax.set_ylim(bottom=0.0)
    _style_axes(ax)

    ax.set_xlim(times[0], times[-1])
    fig.align_ylabels(axes)
    _save(fig, path)


def fig_templates(ex, path):
    """Aligned templates W[:, k] against references r_k, one panel per pitch."""
    freq, W, R, matched = ex["freq"], ex["W"], ex["R"], ex["matched"]
    band = freq <= F_MAX_HZ
    n_pitches = R.shape[1]

    fig, axes = plt.subplots(n_pitches, 1, figsize=(8, 2.4 * n_pitches),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes):
        sim = "" if matched is None else f" (S = {float(matched[k]):.4f})"
        ax.plot(freq[band], R[band, k], color=NEUTRAL, linewidth=2.0,
                label="reference $r_k$")
        ax.plot(freq[band], W[band, k], color=PITCH_COLORS[k], linewidth=1.5,
                linestyle="--", label=f"learned $w_k${sim}")
        ax.set_title(f"{PITCH_NAMES[k]}: learned versus reference template")
        ax.set_ylabel("magnitude\n(column sums to 1)")
        ax.legend(loc="upper right", frameon=False)
        _style_axes(ax)
    axes[-1].set_xlabel("frequency (Hz)")
    axes[-1].set_xlim(0.0, F_MAX_HZ)
    axes[-1].set_ylim(bottom=0.0)
    _save(fig, path)


def fig_f1_by_noise(results, path):
    """Per-clip test F1 with mean and 95% interval, grouped by noise level."""
    cases = [r for r in _read_csv(_require(results / "cases.csv")) if r["metric"] == "f1"]
    summary = [r for r in _read_csv(_require(results / "summary.csv")) if r["metric"] == "f1"]
    if not cases:
        raise SystemExit("make_figures.py: cases.csv has no f1 rows")

    noise_levels = sorted({float(r["noise_std"]) for r in cases})
    methods = method_order({r["method"] for r in cases})
    step = min(0.3, 0.8 / len(methods))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * step
    rng = np.random.default_rng(JITTER_SEED)

    fig, ax = plt.subplots(figsize=(1.8 + (1.6 + 0.4 * len(methods)) * len(noise_levels), 4.2))
    for m, method in enumerate(methods):
        color, marker, name = method_style(method)
        for i, sigma in enumerate(noise_levels):
            values = np.array([float(r["value"]) for r in cases
                               if r["method"] == method and float(r["noise_std"]) == sigma])
            if values.size == 0:
                continue
            x = i + offsets[m]
            half = min(JITTER_HALF_WIDTH, step / 3)
            jitter = rng.uniform(-half, half, size=values.size)
            ax.scatter(x + jitter, values, s=14, color=color, marker=marker, alpha=0.35,
                       linewidths=0, label=f"{name}: per clip" if i == 0 else None)

            row = next((r for r in summary if r["method"] == method
                        and float(r["noise_std"]) == sigma), None)
            if row is None:
                continue
            mean, low, high = float(row["mean"]), float(row["low"]), float(row["high"])
            yerr = None
            if np.isfinite(low) and np.isfinite(high):
                yerr = [[mean - low], [high - mean]]
            ax.errorbar(x + step / 2, mean, yerr=yerr, fmt=marker, color=color, markersize=6,
                        markeredgecolor="white", elinewidth=1.5, capsize=4,
                        label=f"{name}: mean ± 95% CI" if i == 0 else None)

    ax.set_xticks(range(len(noise_levels)), [f"{s:g}" for s in noise_levels])
    ax.set_xlim(-0.6, len(noise_levels) - 0.4)
    ax.set_xlabel("noise standard deviation $\\sigma$ (waveform amplitude units)")
    ax.set_ylabel("frame $F_1$ (unitless)")
    ax.set_title("Test $F_1$ by noise level (one point per clip)")
    ax.set_ylim(top=1.02)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    _style_axes(ax)
    ax.grid(False, axis="x")
    _save(fig, path)


PAIRED_PANELS = [("f1", "$\\Delta$ frame $F_1$ (method − baseline)", "(a) Frame $F_1$, larger is better"),
                 ("onset_mean_abs_lag_frames", "$\\Delta$ mean |onset lag| (frames)",
                  "(b) Onset error, smaller is better")]


def fig_paired(results, path):
    """Per-clip paired differences to the baseline, with mean and 95% interval.

    One panel per metric in ``PAIRED_PANELS`` (separate axes, never a dual axis).
    Points come from ``cases.csv`` (pairs with a NaN on either side are dropped,
    as in ``paired.csv``); means and intervals come from ``paired.csv``.
    """
    paired = _read_csv(_require(results / "paired.csv"))
    cases = _read_csv(_require(results / "cases.csv"))
    value = {(float(r["noise_std"]), r["method"], r["metric"], int(r["seed"])): float(r["value"])
             for r in cases}
    reference = paired[0]["reference"]
    methods = method_order({r["method"] for r in paired})
    noise_levels = sorted({float(r["noise_std"]) for r in paired})
    step = min(0.3, 0.8 / len(methods))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * step
    rng = np.random.default_rng(JITTER_SEED)

    fig, axes = plt.subplots(1, len(PAIRED_PANELS),
                             figsize=(2.0 + (1.3 + 0.35 * len(methods)) * len(noise_levels) * len(PAIRED_PANELS), 4.4))
    for ax, (metric, ylabel, title) in zip(np.atleast_1d(axes), PAIRED_PANELS):
        ax.axhline(0.0, color=NEUTRAL, linewidth=1.0, zorder=1)
        for m, method in enumerate(methods):
            color, marker, name = method_style(method)
            for i, sigma in enumerate(noise_levels):
                diffs = []
                for (n, meth, met, seed), v in value.items():
                    if n == sigma and meth == method and met == metric:
                        ref = value.get((sigma, reference, metric, seed))
                        if ref is not None and np.isfinite(v) and np.isfinite(ref):
                            diffs.append(v - ref)
                x = i + offsets[m]
                if diffs:
                    half = min(JITTER_HALF_WIDTH, step / 3)
                    ax.scatter(x + rng.uniform(-half, half, size=len(diffs)), diffs, s=12,
                               color=color, marker=marker, alpha=0.3, linewidths=0)
                row = next((r for r in paired if r["method"] == method and r["metric"] == metric
                            and float(r["noise_std"]) == sigma), None)
                if row is None or not np.isfinite(float(row["mean_diff"])):
                    continue
                mean, low, high = float(row["mean_diff"]), float(row["low"]), float(row["high"])
                yerr = [[mean - low], [high - mean]] if np.isfinite(low) and np.isfinite(high) else None
                ax.errorbar(x + step / 2, mean, yerr=yerr, fmt=marker, color=color, markersize=6,
                            markeredgecolor="white", elinewidth=1.5, capsize=4,
                            label=f"{name}: mean ± 95% CI" if i == 0 else None)
        ax.set_xticks(range(len(noise_levels)), [f"{s:g}" for s in noise_levels])
        ax.set_xlim(-0.6, len(noise_levels) - 0.4)
        ax.set_xlabel("noise standard deviation $\\sigma$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _style_axes(ax)
        ax.grid(False, axis="x")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Paired differences to the {reference} over test clips (dots: one clip each)")
    _save(fig, path)


VALIDATION_PANELS = [("mu_grid", "mean_f1_by_mu", "penalty weight $\\mu$ (log scale)", True),
                     ("window_grid", "mean_f1_by_window", "post-hoc window $w$ (frames)", False)]


def fig_validation(results, path):
    """Best validation F1 against each tuned setting, one line per noise level.

    One panel per (method, tuned setting); the dashed line in the same colour is
    the baseline's best validation F1 at that noise level. Returns False (and
    writes nothing) if no method was tuned.
    """
    info = json.loads(_require(results / "run_info.json").read_text())
    rows = info.get("validation", [])
    baseline_best = {r["noise_std"]: max(r["mean_f1"]) for r in rows if r.get("method") == "baseline"}
    panels = []
    for grid_key, table_key, xlabel, logx in VALIDATION_PANELS:
        # A one-value grid (e.g. the mu = 0 ablation) has no curve to show.
        tuned = {r["method"] for r in rows if r.get(grid_key) is not None and len(r[grid_key]) > 1}
        for method in method_order(tuned):
            panels.append((method, grid_key, table_key, xlabel, logx))
    if not panels:
        return False

    noise_levels = sorted({r["noise_std"] for r in rows})
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels) + 1.0, 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (method, grid_key, table_key, xlabel, logx) in zip(axes, panels):
        for i, sigma in enumerate(noise_levels):
            row = next((r for r in rows if r["method"] == method and r["noise_std"] == sigma), None)
            if row is None:
                continue
            colour = noise_colour(i, len(noise_levels))
            grid = np.asarray(row[grid_key], dtype=float)
            best = np.max(np.asarray(row[table_key], dtype=float), axis=1)
            ax.plot(grid, best, color=colour, linewidth=2.0, marker="o", markersize=8,
                    markeredgecolor="white", label=f"$\\sigma$ = {sigma:g}")
            if sigma in baseline_best:
                ax.axhline(baseline_best[sigma], color=colour, linestyle="--", linewidth=1.2)
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xticks(grid)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel(xlabel)
        ax.set_title(method_style(method)[2])
        _style_axes(ax)
    axes[0].set_ylabel("best validation $F_1$ (mean over clips)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
                    title="solid: method\ndashed: baseline")
    fig.suptitle("Validation $F_1$ at the best threshold, per tuned setting")
    _save(fig, path)
    return True


def fig_example_methods(ex, path):
    """Scaled activations of the example clip for every method that changes them.

    Returns False (and writes nothing) if no other method was saved.
    """
    others = [k[len("H_scaled__"):] for k in ex if k.startswith("H_scaled__")]
    others = [m for m in method_order(others)
              if not np.array_equal(ex[f"H_scaled__{m}"], ex["H_scaled"])]
    if not others:
        return False
    reference = str(np.asarray(ex.get("reference_method", "baseline")))
    labels = [reference] + others
    times, Y = ex["times"], np.asarray(ex["Y"], dtype=bool)

    fig, axes = plt.subplots(len(labels), 1, figsize=(9, 1.9 * len(labels) + 0.6), sharex=True, sharey=True)
    for ax, label in zip(np.atleast_1d(axes), labels):
        H = ex["H_scaled"] if label == reference else ex[f"H_scaled__{label}"]
        theta = ex["theta"] if label == reference else float(np.asarray(ex[f"theta__{label}"]))
        setting = ""
        if label != reference:
            mu = float(np.asarray(ex.get(f"mu__{label}", np.nan)))
            win = float(np.asarray(ex.get(f"window__{label}", np.nan)))
            setting = f", $\\mu$ = {mu:g}" if np.isfinite(mu) else ""
            setting += f", $w$ = {win:g}" if np.isfinite(win) else ""
        for k in range(H.shape[0]):
            ax.fill_between(times, 0, 1, where=Y[k], step="mid", color=PITCH_COLORS[k], alpha=0.08,
                            linewidth=0, transform=ax.get_xaxis_transform())
            ax.plot(times, H[k], color=PITCH_COLORS[k], linewidth=1.3,
                    label=PITCH_NAMES[k] if label == reference else None)
        ax.axhline(theta, color=NEUTRAL, linestyle="--", linewidth=1.1)
        ax.set_title(f"{method_style(label)[2]} ($\\theta$ = {theta:.2f}{setting})", loc="left", fontsize=10)
        ax.set_ylim(bottom=0.0)
        _style_axes(ax)
    axes = np.atleast_1d(axes)
    axes[0].legend(loc="upper right", ncol=3, frameon=False)
    axes[-1].set_xlabel("time (s)")
    axes[-1].set_xlim(times[0], times[-1])
    fig.supylabel("scaled activation $\\tilde H$ (unitless)")
    fig.suptitle(f"Example validation clip (seed {int(np.asarray(ex.get('seed', -1)))}, "
                 f"$\\sigma$ = {float(np.asarray(ex.get('noise_std', np.nan))):g}); "
                 "shading: true notes, dashed: $\\theta$")
    _save(fig, path)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draw figures from saved results.")
    parser.add_argument("--results", required=True, type=Path,
                        help="results folder written by run_experiments.py")
    args = parser.parse_args(argv)

    results = args.results
    if not results.is_dir():
        raise SystemExit(f"make_figures.py: {results} is not a directory")
    out_dir = results / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    example = load_example(results)
    fig_example(example, out_dir / "fig_example.png")
    fig_templates(example, out_dir / "fig_templates.png")
    fig_f1_by_noise(results, out_dir / "fig_f1_by_noise.png")
    if (results / "paired.csv").is_file():
        fig_paired(results, out_dir / "fig_paired.png")
    fig_validation(results, out_dir / "fig_validation.png")
    fig_example_methods(example, out_dir / "fig_example_methods.png")


if __name__ == "__main__":
    main()
