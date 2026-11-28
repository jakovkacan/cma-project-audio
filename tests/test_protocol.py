"""Section 6.1 of the guide: src/protocol.py (learn_dictionary, evaluate_setting).

Uses a tiny configuration (3 development clips, 2 restarts, 60 iterations), so the
numbers are not meaningful; these tests check the contract and the information flow.
"""
import numbers

import numpy as np
import pytest

from conftest import CASE_COLUMNS, CASE_METRICS, F_BINS, tiny_config, unpack_evaluation

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tiny_dictionary(protocol):
    return protocol.learn_dictionary(tiny_config())


@pytest.fixture(scope="module")
def tiny_evaluation(protocol, tiny_dictionary):
    return unpack_evaluation(protocol.evaluate_setting(tiny_dictionary, 0.05, tiny_config()))


class _SeedRecorder:
    def __init__(self, real):
        self.real, self.calls = real, []

    def __call__(self, seed, *args, **kwargs):
        noise = kwargs.get("noise_std", args[0] if args else 0.005)
        self.calls.append((int(seed), float(noise)))
        return self.real(seed, *args, **kwargs)


def _record_make_clip(monkeypatch, data, protocol):
    rec = _SeedRecorder(data.make_clip)
    monkeypatch.setattr(data, "make_clip", rec)
    if hasattr(protocol, "make_clip"):
        monkeypatch.setattr(protocol, "make_clip", rec)
    return rec


class TestLearnDictionary:
    def test_output_contract(self, tiny_dictionary):
        d = tiny_dictionary
        assert set(d) >= {"W", "scales", "match", "best_seed", "final_losses", "setup_seconds"}
        W = np.asarray(d["W"])
        assert W.shape == (F_BINS, 3)
        assert np.isfinite(W).all() and (W >= 0).all()
        np.testing.assert_allclose(W.sum(axis=0), 1.0, rtol=1e-8)
        scales = np.asarray(d["scales"])
        assert scales.shape == (3,) and (scales > 0).all()
        assert d["best_seed"] in tiny_config()["restart_seeds"]
        assert len(d["final_losses"]) == len(tiny_config()["restart_seeds"])
        assert isinstance(d["setup_seconds"], numbers.Real) and d["setup_seconds"] >= 0

    def test_match_is_a_permutation(self, tiny_dictionary):
        m = tiny_dictionary["match"]
        assert set(m) >= {"perm", "similarity", "matched", "unmatched"}
        assert sorted(np.asarray(m["perm"]).tolist()) == [0, 1, 2]

    def test_best_seed_has_lowest_loss(self, tiny_dictionary):
        seeds = tiny_config()["restart_seeds"]
        assert tiny_dictionary["best_seed"] == seeds[int(np.argmin(tiny_dictionary["final_losses"]))]

    def test_uses_only_development_clips(self, protocol, data, monkeypatch):
        rec = _record_make_clip(monkeypatch, data, protocol)
        cfg = tiny_config()
        protocol.learn_dictionary(cfg)
        assert rec.calls, "learn_dictionary should generate clips with make_clip"
        dev = set(range(*cfg["seeds"]["dev"]))
        used = {s for s, _ in rec.calls}
        assert used <= dev, f"learn_dictionary used non-development seeds: {sorted(used - dev)}"
        assert all(n == pytest.approx(cfg["dev_noise_std"]) for _, n in rec.calls), \
            "development clips must use dev_noise_std"

    def test_deterministic(self, protocol, tiny_dictionary):
        again = protocol.learn_dictionary(tiny_config())
        np.testing.assert_allclose(again["W"], tiny_dictionary["W"], rtol=1e-12)
        np.testing.assert_allclose(again["scales"], tiny_dictionary["scales"], rtol=1e-12)


class TestEvaluateSetting:
    def test_records_contract(self, tiny_evaluation):
        records, threshold = tiny_evaluation
        cfg = tiny_config()
        n_test = len(range(*cfg["seeds"]["test"]))
        assert len(records) == n_test * len(CASE_METRICS), "one record per (test clip, metric)"
        for r in records:
            assert set(CASE_COLUMNS) <= set(r), f"record missing columns: {set(CASE_COLUMNS) - set(r)}"
        assert {r["metric"] for r in records} == CASE_METRICS
        assert {r["split"] for r in records} == {"test"}
        assert {r["method"] for r in records} == {"baseline"}
        assert {int(r["seed"]) for r in records} == set(range(*cfg["seeds"]["test"]))
        assert all(float(r["noise_std"]) == pytest.approx(0.05) for r in records)
        assert all(float(r["runtime_s"]) >= 0 for r in records)

    def test_metric_ranges(self, tiny_evaluation):
        records, _ = tiny_evaluation
        for r in records:
            v = float(r["value"])
            if r["metric"] == "f1":
                assert 0.0 <= v <= 1.0
            elif r["metric"] == "kl_per_entry":
                assert v > 0
            elif r["metric"] == "onsets_missed":
                assert v >= 0 and float(v).is_integer()

    def test_threshold_from_grid(self, tiny_evaluation):
        _, threshold = tiny_evaluation
        g = tiny_config()["threshold_grid"]
        grid = np.arange(g["start"], g["stop"] + g["step"] / 2, g["step"])
        assert np.any(np.isclose(grid, threshold))

    def test_threshold_independent_of_test_clips(self, protocol, tiny_dictionary, tiny_evaluation):
        """Changing only the test seeds must not change theta (it is chosen on validation clips)."""
        cfg = tiny_config(); cfg["seeds"]["test"] = [2000, 2003]
        _, other = unpack_evaluation(protocol.evaluate_setting(tiny_dictionary, 0.05, cfg))
        assert other == pytest.approx(tiny_evaluation[1])

    def test_no_development_seeds_and_noise_level_used(self, protocol, data, tiny_dictionary, monkeypatch):
        rec = _record_make_clip(monkeypatch, data, protocol)
        cfg = tiny_config()
        protocol.evaluate_setting(tiny_dictionary, 0.05, cfg)
        if not rec.calls:
            pytest.skip("evaluate_setting does not call make_clip directly")
        used = {s for s, _ in rec.calls}
        allowed = set(range(*cfg["seeds"]["val"])) | set(range(*cfg["seeds"]["test"]))
        assert used <= allowed, f"unexpected seeds: {sorted(used - allowed)}"
        assert all(n == pytest.approx(0.05) for _, n in rec.calls), "val/test clips must use the setting's noise level"

    def test_deterministic_quality(self, protocol, tiny_dictionary, tiny_evaluation):
        records, _ = unpack_evaluation(protocol.evaluate_setting(tiny_dictionary, 0.05, tiny_config()))
        key = lambda r: (int(r["seed"]), r["metric"])
        a = {key(r): float(r["value"]) for r in tiny_evaluation[0]}
        b = {key(r): float(r["value"]) for r in records}
        assert a.keys() == b.keys()
        for k in a:
            if np.isnan(a[k]):
                assert np.isnan(b[k])
            else:
                assert b[k] == pytest.approx(a[k], rel=1e-10)

    def test_dictionary_not_mutated(self, protocol, tiny_dictionary):
        W0 = np.array(tiny_dictionary["W"], copy=True); s0 = np.array(tiny_dictionary["scales"], copy=True)
        protocol.evaluate_setting(tiny_dictionary, 0.05, tiny_config())
        np.testing.assert_array_equal(tiny_dictionary["W"], W0)
        np.testing.assert_array_equal(tiny_dictionary["scales"], s0)


# ----------------------------------------------------------------------------
# Modified method: evaluate_setting with a mu_grid (joint mu/theta selection)
# ----------------------------------------------------------------------------
MU_GRID = [10.0, 0.0, 1.0]   # deliberately unsorted


@pytest.fixture(scope="module")
def smooth_evaluation(protocol, tiny_dictionary):
    return protocol.evaluate_setting(tiny_dictionary, 0.05, tiny_config(), method="smooth", mu_grid=MU_GRID)


class TestEvaluateSmooth:
    def test_baseline_reports_no_mu(self, protocol, tiny_dictionary):
        out = protocol.evaluate_setting(tiny_dictionary, 0.05, tiny_config())
        assert out["mu"] is None and out["mu_grid"] is None and out["val_mean_f1_by_mu"] is None

    def test_contract(self, smooth_evaluation):
        out = smooth_evaluation
        records = out["records"]
        n_test = len(range(*tiny_config()["seeds"]["test"]))
        assert len(records) == n_test * len(CASE_METRICS)
        assert {r["method"] for r in records} == {"smooth"}
        assert all(r["case_id"].startswith("smooth_") for r in records)
        np.testing.assert_array_equal(out["mu_grid"], sorted(MU_GRID))
        assert out["mu"] in MU_GRID
        assert out["val_mean_f1_by_mu"].shape == (len(MU_GRID), len(out["grid"]))

    def test_selection_is_argmax_with_smallest_mu_on_ties(self, smooth_evaluation):
        out = smooth_evaluation
        best_per_mu = out["val_mean_f1_by_mu"].max(axis=1)
        i = int(np.argmax(best_per_mu))
        assert out["mu"] == out["mu_grid"][i]
        np.testing.assert_array_equal(out["val_mean_f1"], out["val_mean_f1_by_mu"][i])
        assert out["threshold"] == pytest.approx(out["grid"][int(np.argmax(out["val_mean_f1"]))])

    def test_mu_zero_ablation_matches_baseline(self, protocol, tiny_dictionary, tiny_evaluation):
        """mu_grid=[0] runs the modified code path with the penalty off: same theta and quality."""
        out = protocol.evaluate_setting(tiny_dictionary, 0.05, tiny_config(), method="ablation", mu_grid=[0.0])
        base_records, base_theta = tiny_evaluation
        assert out["mu"] == 0.0
        assert out["threshold"] == pytest.approx(base_theta)
        key = lambda r: (int(r["seed"]), r["metric"])
        base = {key(r): r["value"] for r in base_records}
        for r in out["records"]:
            b = base[key(r)]
            assert (np.isnan(b) and np.isnan(r["value"])) or r["value"] == b

    def test_mu_and_threshold_independent_of_test_clips(self, protocol, tiny_dictionary, smooth_evaluation):
        cfg = tiny_config(); cfg["seeds"]["test"] = [2000, 2003]
        other = protocol.evaluate_setting(tiny_dictionary, 0.05, cfg, method="smooth", mu_grid=MU_GRID)
        assert other["mu"] == smooth_evaluation["mu"]
        assert other["threshold"] == pytest.approx(smooth_evaluation["threshold"])
        np.testing.assert_array_equal(other["val_mean_f1_by_mu"], smooth_evaluation["val_mean_f1_by_mu"])

    def test_only_val_and_test_seeds_used(self, protocol, data, tiny_dictionary, monkeypatch):
        rec = _record_make_clip(monkeypatch, data, protocol)
        cfg = tiny_config()
        protocol.evaluate_setting(tiny_dictionary, 0.05, cfg, method="smooth", mu_grid=MU_GRID)
        used = {s for s, _ in rec.calls}
        allowed = set(range(*cfg["seeds"]["val"])) | set(range(*cfg["seeds"]["test"]))
        assert used <= allowed, f"unexpected seeds: {sorted(used - allowed)}"
        assert all(n == pytest.approx(0.05) for _, n in rec.calls)

    def test_dictionary_not_mutated(self, protocol, tiny_dictionary):
        W0 = np.array(tiny_dictionary["W"], copy=True); s0 = np.array(tiny_dictionary["scales"], copy=True)
        protocol.evaluate_setting(tiny_dictionary, 0.05, tiny_config(), method="smooth", mu_grid=[1.0])
        np.testing.assert_array_equal(tiny_dictionary["W"], W0)
        np.testing.assert_array_equal(tiny_dictionary["scales"], s0)

    @pytest.mark.parametrize("bad", [[], [-1.0], [1.0, np.nan], [np.inf]])
    def test_bad_mu_grid_raises(self, protocol, tiny_dictionary, bad):
        with pytest.raises(ValueError):
            protocol.evaluate_setting(tiny_dictionary, 0.05, tiny_config(), method="smooth", mu_grid=bad)
