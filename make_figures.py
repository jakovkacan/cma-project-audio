"""Figures for the baseline milestone

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
                        one group per noise level.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

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
    methods = sorted({r["method"] for r in cases}, key=lambda m: (m != "baseline", m))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * 0.3
    rng = np.random.default_rng(JITTER_SEED)

    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(noise_levels), 4.2))
    for m, method in enumerate(methods):
        color = PITCH_COLORS[m % len(PITCH_COLORS)]
        for i, sigma in enumerate(noise_levels):
            values = np.array([float(r["value"]) for r in cases
                               if r["method"] == method and float(r["noise_std"]) == sigma])
            if values.size == 0:
                continue
            x = i + offsets[m]
            jitter = rng.uniform(-JITTER_HALF_WIDTH, JITTER_HALF_WIDTH, size=values.size)
            ax.scatter(x + jitter, values, s=14, color=color, alpha=0.35, linewidths=0,
                       label=f"{method}: per clip" if i == 0 else None)

            row = next((r for r in summary if r["method"] == method
                        and float(r["noise_std"]) == sigma), None)
            if row is None:
                continue
            mean, low, high = float(row["mean"]), float(row["low"]), float(row["high"])
            yerr = None
            if np.isfinite(low) and np.isfinite(high):
                yerr = [[mean - low], [high - mean]]
            ax.errorbar(x + 0.18, mean, yerr=yerr, fmt="o", color=color, markersize=6,
                        markeredgecolor="white", elinewidth=1.5, capsize=4,
                        label=f"{method}: mean ± 95% CI" if i == 0 else None)

    ax.set_xticks(range(len(noise_levels)), [f"{s:g}" for s in noise_levels])
    ax.set_xlim(-0.6, len(noise_levels) - 0.4)
    ax.set_xlabel("noise standard deviation $\\sigma$ (waveform amplitude units)")
    ax.set_ylabel("frame $F_1$ (unitless)")
    ax.set_title("Test $F_1$ by noise level (one point per clip)")
    ax.set_ylim(top=1.02)
    ax.legend(loc="best", frameon=False)
    _style_axes(ax)
    ax.grid(False, axis="x")
    _save(fig, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draw baseline figures from saved results.")
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


if __name__ == "__main__":
    main()
