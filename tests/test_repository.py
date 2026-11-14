"""Repository rules from the guide's 'Important notes' and Section 6.2 (configs).

These are static checks: required files, pinned dependencies, no external NMF or
plotting inside src/, and configuration values that match the guide.
"""
import json
import re

import pytest

from conftest import REPO_ROOT

REQUIRED = [
    "README.md", "requirements.txt", ".gitignore", "run_experiments.py", "make_figures.py",
    "configs/smoke.json", "configs/baseline.json", "docs/progress.md",
    "src/__init__.py", "src/data.py", "src/baseline.py", "src/alignment.py", "src/metrics.py",
    "src/stats.py", "src/timing.py", "src/protocol.py",
]
CONFIG_KEYS = {
    "name", "seeds", "dev_noise_std", "noise_levels", "n_components", "restart_seeds",
    "n_iter_dict", "n_iter_infer", "eps", "threshold_grid", "scale_percentile",
    "onset_max_lag", "timing", "output_dir",
}


@pytest.mark.parametrize("rel", REQUIRED)
def test_required_file_exists(rel):
    assert (REPO_ROOT / rel).is_file(), f"missing {rel}"


def _requirement_lines(name):
    path = REPO_ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not written yet")
    lines = [l.split("#", 1)[0].strip() for l in path.read_text().splitlines()]
    return [l for l in lines if l]


def test_requirements_pinned():
    lines = [l for l in _requirement_lines("requirements.txt") if not l.startswith("-")]
    assert lines, "requirements.txt is empty"
    unpinned = [l for l in lines if "==" not in l]
    assert not unpinned, f"pin exact versions with '==': {unpinned}"
    names = {re.split(r"[=<>!~\[ ]", l, maxsplit=1)[0].lower() for l in lines}
    assert {"numpy", "scipy", "matplotlib"} <= names
    assert "scikit-learn" not in names and "sklearn" not in names, \
        "an external NMF belongs in requirements-dev.txt only"


def test_dev_requirements_include_main_if_present():
    if not (REPO_ROOT / "requirements-dev.txt").exists():
        pytest.skip("no requirements-dev.txt")
    lines = _requirement_lines("requirements-dev.txt")
    assert any(l.replace(" ", "") == "-rrequirements.txt" for l in lines), \
        "requirements-dev.txt should start with '-r requirements.txt'"


def _src_files():
    files = sorted((REPO_ROOT / "src").glob("*.py"))
    if not files:
        pytest.skip("src/ is empty")
    return files


def test_no_external_nmf_in_src():
    for f in _src_files():
        text = f.read_text()
        assert not re.search(r"^\s*(import|from)\s+sklearn", text, re.M), f"{f.name} imports scikit-learn"
        assert "NMF(" not in text, f"{f.name} appears to use an external NMF class"


def test_no_plotting_in_src():
    for f in _src_files():
        text = f.read_text()
        assert not re.search(r"^\s*(import|from)\s+matplotlib", text, re.M), f"{f.name} imports matplotlib"
        assert "plt.show(" not in text, f"{f.name} calls plt.show()"


def test_gitignore_excludes_venv():
    path = REPO_ROOT / ".gitignore"
    if not path.exists():
        pytest.skip(".gitignore not written yet")
    assert ".venv" in path.read_text()


def _config(name):
    path = REPO_ROOT / "configs" / name
    if not path.exists():
        pytest.skip(f"configs/{name} not written yet")
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", ["smoke.json", "baseline.json"])
def test_config_has_all_keys(name):
    cfg = _config(name)
    assert CONFIG_KEYS <= set(cfg), f"{name} missing keys: {sorted(CONFIG_KEYS - set(cfg))}"
    assert set(cfg["seeds"]) == {"dev", "val", "test"}
    ranges = [set(range(*cfg["seeds"][k])) for k in ("dev", "val", "test")]
    assert all(ranges), "every split needs at least one seed"
    assert ranges[0].isdisjoint(ranges[1]) and ranges[0].isdisjoint(ranges[2]) and ranges[1].isdisjoint(ranges[2])
    assert set(cfg["timing"]) >= {"warmup", "repeats"}
    assert set(cfg["threshold_grid"]) >= {"start", "stop", "step"}


def test_smoke_config_values():
    cfg = _config("smoke.json")
    assert cfg["seeds"] == {"dev": [0, 3], "val": [100, 103], "test": [1000, 1005]}
    assert cfg["n_components"] == 3
    assert cfg["n_iter_dict"] <= 100 and cfg["n_iter_infer"] <= 100, "keep the smoke run fast"


def test_baseline_config_values():
    cfg = _config("baseline.json")
    assert cfg["seeds"] == {"dev": [0, 10], "val": [100, 110], "test": [1000, 1030]}
    assert cfg["noise_levels"] == [0.05, 0.10, 0.20]
    assert cfg["restart_seeds"] == [0, 1, 2, 3]
    assert cfg["n_iter_dict"] == 300 and cfg["n_iter_infer"] == 200
    assert cfg["n_components"] == 3
    assert cfg["dev_noise_std"] == pytest.approx(0.005)
    assert cfg["eps"] == pytest.approx(1e-10)
    assert cfg["scale_percentile"] == pytest.approx(95.0)
    assert cfg["onset_max_lag"] == 8
    assert cfg["threshold_grid"] == {"start": 0.05, "stop": 0.95, "step": 0.05}
    assert cfg["timing"] == {"warmup": 1, "repeats": 5}
    assert cfg["output_dir"] == "results/baseline"
    n_test = len(range(*cfg["seeds"]["test"]))
    assert n_test >= 20, "the brief asks for at least 20 paired cases per setting"
