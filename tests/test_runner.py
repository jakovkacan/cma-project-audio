"""Sections 6.5 and 7 of the guide: run_experiments.py and make_figures.py, end to end.

Both scripts are run in a subprocess from the repository root with a tiny config
whose output directory lives in pytest's temporary folder, so nothing is written
into your results/ directory.
"""
import csv
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from conftest import (CASE_COLUMNS, CASE_METRICS, F_BINS, REPO_ROOT, SUMMARY_COLUMNS,
                      T_FRAMES, tiny_config)

pytestmark = pytest.mark.integration

RUNNER = REPO_ROOT / "run_experiments.py"
FIGURES = REPO_ROOT / "make_figures.py"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _run(args, timeout=600):
    return subprocess.run([sys.executable, *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="module")
def run_output(tmp_path_factory):
    if not RUNNER.exists():
        pytest.skip("run_experiments.py not written yet")
    tmp = tmp_path_factory.mktemp("runner")
    out_dir = tmp / "nested" / "out"                      # must be created by the runner
    cfg = tiny_config(str(out_dir))
    cfg["noise_levels"] = [0.05, 0.2]
    cfg_path = tmp / "tiny.json"
    cfg_path.write_text(json.dumps(cfg))
    proc = _run([str(RUNNER), "--config", str(cfg_path)])
    assert proc.returncode == 0, f"run_experiments.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return cfg, out_dir


def _read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


class TestRunExperiments:
    def test_creates_outputs(self, run_output):
        _, out = run_output
        for name in ("cases.csv", "summary.csv", "run_info.json", "example_clip.npz"):
            assert (out / name).is_file(), f"missing {name}"

    def test_cases_csv(self, run_output):
        cfg, out = run_output
        header, rows = _read_csv(out / "cases.csv")
        assert set(header) == set(CASE_COLUMNS), f"columns must be {CASE_COLUMNS}"
        n_test = len(range(*cfg["seeds"]["test"]))
        assert len(rows) == len(cfg["noise_levels"]) * n_test * len(CASE_METRICS)
        assert {r["metric"] for r in rows} == CASE_METRICS
        assert {r["split"] for r in rows} == {"test"}
        assert {r["method"] for r in rows} == {"baseline"}
        assert {int(r["seed"]) for r in rows} == set(range(*cfg["seeds"]["test"]))
        assert {float(r["noise_std"]) for r in rows} == set(cfg["noise_levels"])
        for r in rows:
            assert float(r["runtime_s"]) >= 0
            if r["metric"] == "f1":
                assert 0 <= float(r["value"]) <= 1

    def test_summary_csv(self, run_output):
        cfg, out = run_output
        header, rows = _read_csv(out / "summary.csv")
        assert SUMMARY_COLUMNS <= set(header), f"missing columns {SUMMARY_COLUMNS - set(header)}"
        assert len(rows) == len(cfg["noise_levels"]) * len(CASE_METRICS)
        n_test = len(range(*cfg["seeds"]["test"]))
        for r in rows:
            assert int(float(r["n"])) == n_test, "n is the number of test clips"
            lo, m, hi = float(r["low"]), float(r["mean"]), float(r["high"])
            if not any(math.isnan(x) for x in (lo, m, hi)):
                assert lo <= m + 1e-12 and m <= hi + 1e-12

    def test_summary_matches_cases(self, run_output):
        _, out = run_output
        _, cases = _read_csv(out / "cases.csv")
        _, summary = _read_csv(out / "summary.csv")
        for s in summary:
            if s["metric"] != "f1":
                continue
            vals = [float(c["value"]) for c in cases
                    if c["metric"] == "f1" and float(c["noise_std"]) == float(s["noise_std"])
                    and c["method"] == s["method"]]
            assert float(s["mean"]) == pytest.approx(np.mean(vals), rel=1e-9)

    def test_run_info(self, run_output):
        cfg, out = run_output
        text = (out / "run_info.json").read_text()
        info = json.loads(text)
        import platform
        import scipy
        assert np.__version__ in text, "record the NumPy version"
        assert scipy.__version__ in text, "record the SciPy version"
        assert platform.python_version() in text, "record the Python version"

        def contains_config(obj):
            if isinstance(obj, dict):
                if obj.get("restart_seeds") == cfg["restart_seeds"] and obj.get("seeds") == cfg["seeds"]:
                    return True
                return any(contains_config(v) for v in obj.values())
            if isinstance(obj, list):
                return any(contains_config(v) for v in obj)
            return False

        assert contains_config(info), "run_info.json must contain the full configuration"

    def test_example_clip(self, run_output):
        _, out = run_output
        with np.load(out / "example_clip.npz") as z:
            assert {"V", "Y"} <= set(z.files), f"example_clip.npz must contain V and Y (has {z.files})"
            assert z["V"].shape == (F_BINS, T_FRAMES)
            assert z["Y"].shape == (3, T_FRAMES)
            assert len(z.files) >= 5, "store V, Y, scaled H, theta and the time/frequency axes"

    def test_missing_config_fails(self):
        if not RUNNER.exists():
            pytest.skip("run_experiments.py not written yet")
        assert _run([str(RUNNER), "--config", str(REPO_ROOT / "configs" / "does_not_exist.json")]).returncode != 0


def test_baseline_only_run_writes_no_paired_csv(run_output):
    _, out = run_output
    assert not (out / "paired.csv").exists()


# ----------------------------------------------------------------------------
# Baseline vs modified in one run ("methods" in the config)
# ----------------------------------------------------------------------------
METHODS = [{"label": "baseline"}, {"label": "smooth", "mu_grid": [1.0, 10.0]},
           {"label": "smooth_mu0", "mu_grid": [0.0]},
           {"label": "posthoc_median", "window_grid": [3, 5], "filter": "median"}]
TUNED = {"smooth", "smooth_mu0", "posthoc_median"}
PAIRED_COLUMNS = ["noise_std", "method", "reference", "metric", "mean_diff", "low", "high", "sd", "n"]


@pytest.fixture(scope="module")
def methods_output(tmp_path_factory):
    if not RUNNER.exists():
        pytest.skip("run_experiments.py not written yet")
    tmp = tmp_path_factory.mktemp("runner_methods")
    out_dir = tmp / "out"
    cfg = tiny_config(str(out_dir))
    cfg["noise_levels"] = [0.05, 0.2]
    cfg["methods"] = METHODS
    cfg_path = tmp / "tiny_methods.json"
    cfg_path.write_text(json.dumps(cfg))
    proc = _run([str(RUNNER), "--config", str(cfg_path)])
    assert proc.returncode == 0, f"run_experiments.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return cfg, out_dir


class TestRunMethods:
    def test_cases_cover_every_method_and_pair(self, methods_output):
        cfg, out = methods_output
        _, rows = _read_csv(out / "cases.csv")
        n_test = len(range(*cfg["seeds"]["test"]))
        labels = {m["label"] for m in METHODS}
        assert {r["method"] for r in rows} == labels
        assert len(rows) == len(labels) * len(cfg["noise_levels"]) * n_test * len(CASE_METRICS)
        for label in labels:
            seeds = {int(r["seed"]) for r in rows if r["method"] == label}
            assert seeds == set(range(*cfg["seeds"]["test"])), "every method sees the same test clips"

    def test_summary_rows_per_method(self, methods_output):
        cfg, out = methods_output
        _, rows = _read_csv(out / "summary.csv")
        assert len(rows) == len(METHODS) * len(cfg["noise_levels"]) * len(CASE_METRICS)

    def test_paired_csv(self, methods_output):
        cfg, out = methods_output
        header, rows = _read_csv(out / "paired.csv")
        assert header == PAIRED_COLUMNS
        assert {r["method"] for r in rows} == TUNED
        assert {r["reference"] for r in rows} == {"baseline"}
        assert {r["metric"] for r in rows} == CASE_METRICS | {"runtime_s"}
        assert len(rows) == len(TUNED) * len(cfg["noise_levels"]) * (len(CASE_METRICS) + 1)

    def test_paired_matches_cases(self, methods_output):
        _, out = methods_output
        _, cases = _read_csv(out / "cases.csv")
        _, paired = _read_csv(out / "paired.csv")
        val = {(float(c["noise_std"]), c["method"], int(c["seed"])): float(c["value"])
               for c in cases if c["metric"] == "f1"}
        for p in paired:
            if p["metric"] != "f1":
                continue
            sigma = float(p["noise_std"])
            diffs = [v - val[(sigma, "baseline", s)] for (n, m, s), v in val.items()
                     if n == sigma and m == p["method"]]
            assert float(p["mean_diff"]) == pytest.approx(np.mean(diffs), abs=1e-12)

    def test_ablation_differences_are_zero(self, methods_output):
        """smooth_mu0 runs the modified code with the penalty off: every quality difference is 0."""
        _, out = methods_output
        _, paired = _read_csv(out / "paired.csv")
        for p in paired:
            if p["method"] == "smooth_mu0" and p["metric"] != "runtime_s":
                assert float(p["mean_diff"]) == 0.0, p

    def test_run_info_records_mu(self, methods_output):
        cfg, out = methods_output
        info = json.loads((out / "run_info.json").read_text())
        rows = {(t["noise_std"], t["method"]): t for t in info["thresholds"]}
        assert len(rows) == len(METHODS) * len(cfg["noise_levels"])
        for sigma in cfg["noise_levels"]:
            assert rows[(sigma, "baseline")]["mu"] is None
            assert rows[(sigma, "smooth")]["mu"] in (1.0, 10.0)
            assert rows[(sigma, "smooth_mu0")]["mu"] == 0.0
            assert rows[(sigma, "posthoc_median")]["window"] in (3, 5)
            assert rows[(sigma, "posthoc_median")]["filter"] == "median"
        for v in info["validation"]:
            if v["method"] == "smooth":
                assert np.asarray(v["mean_f1_by_mu"]).shape == (2, len(v["grid"]))

    def test_example_clip_has_every_method(self, methods_output):
        _, out = methods_output
        with np.load(out / "example_clip.npz") as z:
            assert str(z["reference_method"]) == "baseline"
            for label in TUNED:
                assert z[f"H_scaled__{label}"].shape == z["H_scaled"].shape
            np.testing.assert_array_equal(z["H_scaled__smooth_mu0"], z["H_scaled"])

    def test_duplicate_labels_fail(self, tmp_path):
        cfg = tiny_config(str(tmp_path / "out"))
        cfg["methods"] = [{"label": "baseline"}, {"label": "baseline", "mu_grid": [1.0]}]
        path = tmp_path / "dup.json"; path.write_text(json.dumps(cfg))
        assert _run([str(RUNNER), "--config", str(path)]).returncode != 0

    def test_figures_from_methods_run(self, methods_output):
        if not FIGURES.exists():
            pytest.skip("make_figures.py not written yet")
        _, out = methods_output
        proc = _run([str(FIGURES), "--results", str(out)], timeout=300)
        assert proc.returncode == 0, f"make_figures.py failed:\n{proc.stderr}"
        for name in ("fig_example.png", "fig_templates.png", "fig_f1_by_noise.png", "fig_paired.png",
                     "fig_validation.png", "fig_example_methods.png"):
            path = out / "figures" / name
            assert path.is_file(), f"missing figures/{name}"
            assert path.read_bytes()[:8] == PNG_MAGIC

    def test_both_grids_fail(self, tmp_path):
        cfg = tiny_config(str(tmp_path / "out"))
        cfg["methods"] = [{"label": "baseline"}, {"label": "x", "mu_grid": [1.0], "window_grid": [3]}]
        path = tmp_path / "both.json"; path.write_text(json.dumps(cfg))
        assert _run([str(RUNNER), "--config", str(path)]).returncode != 0


class TestMakeFigures:
    def test_writes_three_pngs(self, run_output):
        if not FIGURES.exists():
            pytest.skip("make_figures.py not written yet")
        _, out = run_output
        proc = _run([str(FIGURES), "--results", str(out)], timeout=300)
        assert proc.returncode == 0, f"make_figures.py failed:\n{proc.stderr}"
        for name in ("fig_example.png", "fig_templates.png", "fig_f1_by_noise.png"):
            path = out / "figures" / name
            assert path.is_file(), f"missing figures/{name}"
        for name in ("fig_paired.png", "fig_validation.png", "fig_example_methods.png"):
            assert not (out / "figures" / name).exists(), f"baseline-only run should not write {name}"
            assert path.read_bytes()[:8] == PNG_MAGIC, f"{name} is not a PNG"
            assert path.stat().st_size > 1000
