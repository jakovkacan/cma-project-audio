"""Sections 6.3-6.4 of the guide: src/stats.py (mean_ci) and src/timing.py (time_call)."""
import time

import numpy as np
import pytest
from scipy import stats as sps

from conftest import is_integer


class TestMeanCI:
    def test_known_values(self, stats):
        out = stats.mean_ci([1, 2, 3, 4, 5])
        assert set(out) >= {"mean", "low", "high", "sd", "n"}
        half = sps.t.ppf(0.975, 4) * np.sqrt(2.5) / np.sqrt(5)       # 2.7764 * 1.5811 / 2.2361
        assert out["mean"] == pytest.approx(3.0)
        assert out["sd"] == pytest.approx(np.sqrt(2.5)), "use the sample s.d. (ddof=1)"
        assert out["low"] == pytest.approx(3.0 - half) and out["high"] == pytest.approx(3.0 + half)
        assert out["low"] == pytest.approx(1.0368, abs=1e-4)
        assert is_integer(out["n"]) and out["n"] == 5

    def test_random_data_against_formula(self, stats, rng):
        x = rng.normal(0.8, 0.1, size=30)
        out = stats.mean_ci(x)
        half = sps.t.ppf(0.975, 29) * x.std(ddof=1) / np.sqrt(30)
        assert out["low"] == pytest.approx(x.mean() - half, rel=1e-10)
        assert out["high"] == pytest.approx(x.mean() + half, rel=1e-10)

    def test_level_changes_width(self, stats, rng):
        x = rng.normal(size=20)
        w95 = stats.mean_ci(x)["high"] - stats.mean_ci(x)["low"]
        w90 = stats.mean_ci(x, level=0.90)["high"] - stats.mean_ci(x, level=0.90)["low"]
        assert w90 < w95
        half90 = sps.t.ppf(0.95, 19) * x.std(ddof=1) / np.sqrt(20)
        assert w90 == pytest.approx(2 * half90, rel=1e-10)

    def test_constant_values(self, stats):
        out = stats.mean_ci([0.9] * 10)
        assert out["sd"] == pytest.approx(0.0, abs=1e-15)
        assert out["low"] == pytest.approx(0.9) and out["high"] == pytest.approx(0.9)

    def test_input_not_mutated(self, stats):
        x = np.array([3.0, 1.0, 2.0]); x0 = x.copy()
        stats.mean_ci(x)
        np.testing.assert_array_equal(x, x0)


class TestTimeCall:
    def test_counts_warmup_and_repeats(self, timing):
        calls = []

        def fn(a, b=0):
            calls.append((a, b))
            return len(calls)

        out = timing.time_call(fn, 1, b=2, warmup=2, repeats=4)
        assert set(out) >= {"median_s", "min_s", "all_s", "result"}
        assert len(calls) == 6, "warmup calls + timed calls"
        assert all(c == (1, 2) for c in calls), "positional and keyword arguments must be passed through"
        assert len(out["all_s"]) == 4
        assert out["result"] == 6, "'result' must be the return value of the last call"

    def test_summary_consistent(self, timing):
        out = timing.time_call(lambda: sum(range(1000)), warmup=1, repeats=5)
        all_s = np.asarray(out["all_s"])
        assert (all_s >= 0).all()
        assert out["median_s"] == pytest.approx(float(np.median(all_s)))
        assert out["min_s"] == pytest.approx(float(np.min(all_s)))

    def test_measures_wall_time(self, timing):
        out = timing.time_call(time.sleep, 0.01, warmup=0, repeats=3)
        assert out["min_s"] >= 0.009
        assert out["result"] is None

    def test_defaults(self, timing):
        calls = []
        timing.time_call(lambda: calls.append(1))
        assert len(calls) == 6, "defaults are warmup=1, repeats=5"
