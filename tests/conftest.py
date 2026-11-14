"""Shared fixtures for the Track A baseline test suite (Project Guide A1).

Run from the repository root:

    python -m pytest                    # everything
    python -m pytest -m "not slow"      # quick loop while developing
    python -m pytest -rs                # also list skipped tests and why

A module that does not exist yet (for example ``src/stats.py`` before you write it)
makes its tests *skip* rather than fail, so you can implement section by section.
Any other import problem (a typo, a missing dependency inside your module) is an
error, not a skip. ``tests/test_repository.py`` fails if a required file is missing,
so a fully skipped module still shows up before submission.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Shapes fixed by the guide's generator (fs = 8000 Hz, 6 s, nperseg = 512, hop = 128).
F_BINS = 257
T_FRAMES = 372
N_PITCHES = 3

# Columns and metric names fixed by Section 6.5 of the guide.
CASE_COLUMNS = ["case_id", "split", "seed", "noise_std", "method", "metric", "value", "runtime_s"]
CASE_METRICS = {"f1", "kl_per_entry", "onset_mean_abs_lag_frames", "onsets_missed"}
SUMMARY_COLUMNS = {"noise_std", "method", "metric", "mean", "low", "high", "sd", "n"}


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: reference-value regression tests that run the full development-set "
        "dictionary learning (about 1 minute). Deselect with -m 'not slow'.",
    )
    config.addinivalue_line(
        "markers", "integration: runs scripts end to end in a subprocess."
    )


def _import_or_skip(name: str):
    """Import ``name``; skip only if that exact module does not exist yet."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name in {name, name.rsplit(".", 1)[0]}:
            pytest.skip(f"{name} not implemented yet ({exc})")
        raise


# ----------------------------------------------------------------------------
# Module fixtures: tests receive your module and call functions on it.
# ----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def data():
    return _import_or_skip("src.data")


@pytest.fixture(scope="session")
def baseline():
    return _import_or_skip("src.baseline")


@pytest.fixture(scope="session")
def alignment():
    return _import_or_skip("src.alignment")


@pytest.fixture(scope="session")
def metrics():
    return _import_or_skip("src.metrics")


@pytest.fixture(scope="session")
def stats():
    return _import_or_skip("src.stats")


@pytest.fixture(scope="session")
def timing():
    return _import_or_skip("src.timing")


@pytest.fixture(scope="session")
def protocol():
    return _import_or_skip("src.protocol")


# ----------------------------------------------------------------------------
# Data fixtures
# ----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def _clip0_cached(data):
    return data.make_clip(0)


@pytest.fixture
def clip0(_clip0_cached):
    """make_clip(0), copied per test so one test cannot corrupt another."""
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in _clip0_cached.items()}


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def positive_V(rng):
    """A small strictly positive matrix (F=12, T=20)."""
    return rng.uniform(0.1, 2.0, size=(12, 20))


@pytest.fixture
def hand_V():
    """The hand-computed example of Section 3.3."""
    return np.array([[1.0, 2.0], [3.0, 4.0]])


def tiny_config(output_dir: str | None = None) -> dict:
    """A very small configuration with the same keys as configs/smoke.json."""
    return {
        "name": "pytest-tiny",
        "seeds": {"dev": [0, 3], "val": [100, 102], "test": [1000, 1003]},
        "dev_noise_std": 0.005,
        "noise_levels": [0.05],
        "n_components": 3,
        "restart_seeds": [0, 1],
        "n_iter_dict": 60,
        "n_iter_infer": 60,
        "eps": 1e-10,
        "threshold_grid": {"start": 0.05, "stop": 0.95, "step": 0.05},
        "scale_percentile": 95.0,
        "onset_max_lag": 8,
        "timing": {"warmup": 0, "repeats": 1},
        "output_dir": output_dir or "results/pytest-tiny",
    }


@pytest.fixture
def tiny_cfg():
    return tiny_config()


def unpack_evaluation(out):
    """Accept either convention for evaluate_setting's return value.

    The guide says it returns "a list of per-case records plus the selected
    threshold". Both ``{"records": [...], "threshold": t, ...}`` and
    ``(records, threshold)`` are accepted; a dictionary is preferred.
    """
    if isinstance(out, dict):
        assert "records" in out and "threshold" in out, (
            "evaluate_setting returned a dict without 'records' and 'threshold' keys"
        )
        return out["records"], out["threshold"]
    records, threshold = out
    return records, threshold


def is_integer(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
