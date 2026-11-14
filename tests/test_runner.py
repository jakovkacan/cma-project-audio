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
                    if c["metric"] == "f1" and float(c["noise_std"]) == float(s["noise_std"])]
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
            assert path.read_bytes()[:8] == PNG_MAGIC, f"{name} is not a PNG"
            assert path.stat().st_size > 1000
