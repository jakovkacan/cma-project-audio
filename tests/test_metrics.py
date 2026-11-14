"""Section 5 of the guide: src/metrics.py (frame F1, KL per entry, onset lags, threshold)."""
import numbers

import numpy as np
import pytest

from conftest import is_integer


def B(rows):
    return np.array(rows, dtype=bool)


class TestFrameF1:
    def test_hand_example(self, metrics):
        out = metrics.frame_f1(B([[1, 1, 0, 0], [0, 1, 1, 0]]), B([[1, 0, 0, 0], [0, 1, 1, 1]]))
        assert set(out) >= {"f1", "tp", "fp", "fn"}
        assert (out["tp"], out["fp"], out["fn"]) == (3, 1, 1)
        assert out["f1"] == pytest.approx(0.75)

    def test_types(self, metrics):
        out = metrics.frame_f1(B([[1, 0]]), B([[1, 1]]))
        assert isinstance(out["f1"], float)
        for k in ("tp", "fp", "fn"):
            assert isinstance(out[k], numbers.Integral) and not isinstance(out[k], bool)

    def test_empty_is_zero(self, metrics):
        out = metrics.frame_f1(np.zeros((3, 5), bool), np.zeros((3, 5), bool))
        assert out["f1"] == 0.0 and (out["tp"], out["fp"], out["fn"]) == (0, 0, 0)

    def test_perfect_and_disjoint(self, metrics, rng):
        Y = rng.uniform(size=(3, 50)) > 0.6
        assert metrics.frame_f1(Y, Y)["f1"] == pytest.approx(1.0)
        assert metrics.frame_f1(~Y, Y)["f1"] == pytest.approx(0.0)

    def test_matches_formula(self, metrics, rng):
        P = rng.uniform(size=(3, 40)) > 0.5; Y = rng.uniform(size=(3, 40)) > 0.5
        tp = np.sum(P & Y); fp = np.sum(P & ~Y); fn = np.sum(~P & Y)
        assert metrics.frame_f1(P, Y)["f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn))


class TestKLPerEntry:
    def test_is_kl_divided_by_size(self, metrics, baseline, rng):
        V = rng.uniform(0, 1, (6, 9)); W = rng.uniform(0.1, 1, (6, 2)); H = rng.uniform(0.1, 1, (2, 9))
        d = metrics.kl_per_entry(V, W, H)
        assert isinstance(d, float)
        assert d == pytest.approx(baseline.kl_divergence(V, W @ H) / V.size, rel=1e-12)


class TestOnsetLags:
    def _one(self, metrics, pred, true, **kw):
        return metrics.onset_lags(B([pred]), B([true]), **kw)

    def test_late_by_one_frame(self, metrics):
        out = self._one(metrics, [0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0])
        assert set(out) >= {"lags", "missed", "n_onsets"}
        np.testing.assert_array_equal(out["lags"], [1])
        assert out["missed"] == 0 and out["n_onsets"] == 1
        assert np.issubdtype(np.asarray(out["lags"]).dtype, np.integer)
        assert is_integer(out["missed"]) and is_integer(out["n_onsets"])

    def test_early_is_negative(self, metrics):
        np.testing.assert_array_equal(self._one(metrics, [0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0])["lags"], [-1])

    def test_onset_at_first_frame_counts(self, metrics):
        out = self._one(metrics, [1, 1, 0, 0], [1, 1, 0, 0])
        assert out["n_onsets"] == 1
        np.testing.assert_array_equal(out["lags"], [0])

    def test_missed_onset(self, metrics):
        out = self._one(metrics, [0] * 8, [0, 0, 1, 1, 0, 0, 0, 0])
        assert out["missed"] == 1 and out["n_onsets"] == 1 and np.asarray(out["lags"]).size == 0

    def test_window_respects_max_lag(self, metrics):
        true = [0] * 20; true[10] = true[11] = 1
        pred = [0] * 20; pred[13] = 1
        assert self._one(metrics, pred, true, max_lag=2)["missed"] == 1
        np.testing.assert_array_equal(self._one(metrics, pred, true, max_lag=3)["lags"], [3])

    def test_window_clipped_at_start(self, metrics):
        out = self._one(metrics, [1] * 12, [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], max_lag=8)
        np.testing.assert_array_equal(out["lags"], [-2])

    def test_default_max_lag_is_8(self, metrics):
        true = [0] * 30; true[10] = 1
        pred = [0] * 30; pred[18] = 1
        np.testing.assert_array_equal(self._one(metrics, pred, true)["lags"], [8])
        pred = [0] * 30; pred[19] = 1
        assert self._one(metrics, pred, true)["missed"] == 1

    def test_several_pitches_and_notes(self, metrics):
        true = B([[1, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0, 0]])
        pred = B([[1, 1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0]])
        out = metrics.onset_lags(true=true, pred=pred, max_lag=1)
        assert out["n_onsets"] == 3 and out["missed"] == 1
        np.testing.assert_array_equal(sorted(out["lags"]), [0, 1])

    def test_no_onsets(self, metrics):
        out = metrics.onset_lags(np.ones((2, 5), bool), np.zeros((2, 5), bool))
        assert out["n_onsets"] == 0 and out["missed"] == 0 and np.asarray(out["lags"]).size == 0


class TestSelectThreshold:
    GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)

    def test_mean_of_per_clip_f1(self, metrics, rng):
        # Clips of very different sizes: pooling frames would give different numbers.
        H_list, Y_list = [], []
        for T in (5, 80, 300):
            Y = rng.uniform(size=(3, T)) > 0.5
            H_list.append(np.where(Y, rng.uniform(0.3, 1.2, (3, T)), rng.uniform(0, 0.6, (3, T))))
            Y_list.append(Y)
        out = metrics.select_threshold(H_list, Y_list, self.GRID)
        assert set(out) >= {"threshold", "grid", "mean_f1"}
        expected = [np.mean([metrics.frame_f1(h > t, y)["f1"] for h, y in zip(H_list, Y_list)]) for t in self.GRID]
        np.testing.assert_allclose(out["mean_f1"], expected, rtol=1e-12, err_msg="average F1 per clip; do not pool frames")
        np.testing.assert_allclose(out["grid"], self.GRID)
        assert out["threshold"] == pytest.approx(self.GRID[int(np.argmax(expected))])

    def test_known_best_threshold(self, metrics):
        Y = B([[0, 1, 1, 0, 1]])
        H = np.array([[0.2, 0.7, 0.9, 0.4, 0.6]])
        out = metrics.select_threshold([H], [Y], [0.1, 0.3, 0.5, 0.8])
        assert out["threshold"] == pytest.approx(0.5)

    def test_ties_break_to_smallest(self, metrics):
        Y = B([[0, 1]]); H = np.array([[0.0, 1.0]])
        out = metrics.select_threshold([H], [Y], [0.2, 0.4, 0.6])     # all perfect
        assert out["threshold"] == pytest.approx(0.2)

    def test_threshold_is_a_grid_value(self, metrics, rng):
        H = [rng.uniform(0, 1, (3, 40))]; Y = [rng.uniform(size=(3, 40)) > 0.5]
        out = metrics.select_threshold(H, Y, self.GRID)
        assert np.any(np.isclose(self.GRID, out["threshold"]))
