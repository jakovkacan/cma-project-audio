"""Section 3 of the guide: src/baseline.py (KL-NMF with multiplicative updates)."""
import math

import numpy as np
import pytest
from scipy.special import kl_div

from conftest import F_BINS, T_FRAMES, is_integer

FIT_KEYS = {"W", "H", "loss", "seed", "n_iter"}


def spec_init(V, K, seed):
    """Initialization exactly as specified in Section 3.2 (Eq. 9)."""
    F, T = V.shape
    rng = np.random.default_rng(seed)
    W0 = rng.uniform(0.5, 1.5, size=(F, K))
    H0 = rng.uniform(0.5, 1.5, size=(K, T))
    alpha = math.sqrt(V.mean() / (W0 @ H0).mean())
    return alpha * W0, alpha * H0


def spec_infer_init(V, W, seed):
    rng = np.random.default_rng(seed)
    H0 = rng.uniform(0.5, 1.5, size=(W.shape[1], V.shape[1]))
    return H0 * V.mean() / (W @ H0).mean()


def nonincreasing(loss, rtol=1e-9):
    loss = np.asarray(loss)
    return bool(np.all(np.diff(loss) <= rtol * np.abs(loss[:-1]) + 1e-15))


# ----------------------------------------------------------------------------
# kl_divergence (Eq. 5)
# ----------------------------------------------------------------------------
class TestKLDivergence:
    def test_hand_value(self, baseline, hand_V):
        # sum(V log V - V + 1) = 2 ln 2 + 3 ln 3 + 4 ln 4 - 6
        expected = 2 * math.log(2) + 3 * math.log(3) + 4 * math.log(4) - 6
        assert baseline.kl_divergence(hand_V, np.ones((2, 2))) == pytest.approx(expected, rel=1e-12)
        assert expected == pytest.approx(4.22731, abs=1e-5)

    def test_returns_python_float(self, baseline, positive_V):
        d = baseline.kl_divergence(positive_V, positive_V + 0.1)
        assert isinstance(d, float)

    def test_zero_iff_equal(self, baseline, positive_V):
        assert baseline.kl_divergence(positive_V, positive_V) == pytest.approx(0.0, abs=1e-12)
        assert baseline.kl_divergence(positive_V, 1.1 * positive_V) > 0

    def test_matches_scipy_elementwise_definition(self, baseline, rng):
        V = rng.uniform(0, 1, size=(9, 11))
        Z = rng.uniform(0.05, 1, size=(9, 11))
        assert baseline.kl_divergence(V, Z) == pytest.approx(float(kl_div(V, Z).sum()), rel=1e-12)

    def test_zero_log_zero_convention(self, baseline):
        # 0 log(0/2) - 0 + 2  +  1 log(1/1) - 1 + 1  = 2
        assert baseline.kl_divergence(np.array([[0.0, 1.0]]), np.array([[2.0, 1.0]])) == pytest.approx(2.0)

    def test_zero_prediction_is_floored(self, baseline):
        d = baseline.kl_divergence(np.array([[1.0]]), np.array([[0.0]]), eps=1e-10)
        assert math.isfinite(d)
        assert d == pytest.approx(math.log(1e10) - 1 + 1e-10, rel=1e-9)

    def test_scale_behaviour(self, baseline, rng):
        # D(aV || aZ) = a D(V || Z)  (Appendix B)
        V = rng.uniform(0.1, 1, (5, 6)); Z = rng.uniform(0.1, 1, (5, 6))
        assert baseline.kl_divergence(3 * V, 3 * Z) == pytest.approx(3 * baseline.kl_divergence(V, Z), rel=1e-10)

    def test_shape_mismatch_raises(self, baseline):
        with pytest.raises(ValueError):
            baseline.kl_divergence(np.ones((2, 3)), np.ones((3, 2)))

    def test_negative_V_raises(self, baseline):
        with pytest.raises(ValueError):
            baseline.kl_divergence(np.array([[1.0, -0.1]]), np.ones((1, 2)))

    def test_inputs_not_mutated(self, baseline, positive_V):
        V = positive_V.copy(); Z = np.zeros_like(V)
        baseline.kl_divergence(V, Z)
        np.testing.assert_array_equal(V, positive_V)
        np.testing.assert_array_equal(Z, 0.0)


# ----------------------------------------------------------------------------
# init_factors (Eq. 9)
# ----------------------------------------------------------------------------
class TestInitFactors:
    def test_keys_shapes_positive(self, baseline, positive_V):
        out = baseline.init_factors(positive_V, 4, seed=0)
        assert set(out) == {"W", "H"}
        assert out["W"].shape == (12, 4) and out["H"].shape == (4, 20)
        assert (out["W"] > 0).all() and (out["H"] > 0).all()

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_follows_specification_exactly(self, baseline, positive_V, seed):
        W, H = spec_init(positive_V, 3, seed)
        out = baseline.init_factors(positive_V, 3, seed)
        np.testing.assert_allclose(out["W"], W, rtol=1e-12, err_msg=(
            "Draw W ~ U(0.5,1.5) first, then H, from np.random.default_rng(seed); "
            "then rescale both by alpha = sqrt(mean V / mean(W0 H0))."))
        np.testing.assert_allclose(out["H"], H, rtol=1e-12)

    def test_reference_first_row(self, baseline):
        out = baseline.init_factors(np.ones((4, 5)), 2, seed=0)
        np.testing.assert_allclose(out["W"][0], [0.7827, 0.5299], atol=1e-4)

    def test_mean_matches_V(self, baseline, positive_V):
        out = baseline.init_factors(positive_V, 3, seed=5)
        assert (out["W"] @ out["H"]).mean() == pytest.approx(positive_V.mean(), rel=1e-12)

    def test_seed_controls_draw(self, baseline, positive_V):
        a = baseline.init_factors(positive_V, 3, 0)
        b = baseline.init_factors(positive_V, 3, 0)
        c = baseline.init_factors(positive_V, 3, 1)
        np.testing.assert_array_equal(a["W"], b["W"])
        assert not np.allclose(a["W"], c["W"])


# ----------------------------------------------------------------------------
# update_activations / update_templates (Eqs. 6-7)
# ----------------------------------------------------------------------------
class TestUpdates:
    def test_hand_activation_update(self, baseline, hand_V):
        H1 = baseline.update_activations(hand_V, np.ones((2, 1)), np.ones((1, 2)))
        np.testing.assert_allclose(H1, [[2.0, 3.0]], atol=1e-12)

    def test_hand_template_update(self, baseline, hand_V):
        # Q must be recomputed from W0 H1, not reused from the H step.
        W1 = baseline.update_templates(hand_V, np.ones((2, 1)), np.array([[2.0, 3.0]]))
        np.testing.assert_allclose(W1, [[0.6], [1.4]], atol=1e-12)

    def test_matrix_form(self, baseline, rng):
        V = rng.uniform(0.1, 1, (7, 9)); W = rng.uniform(0.1, 1, (7, 3)); H = rng.uniform(0.1, 1, (3, 9))
        Q = V / (W @ H)
        np.testing.assert_allclose(baseline.update_activations(V, W, H),
                                   H * (W.T @ Q) / W.sum(axis=0)[:, None], rtol=1e-12)
        np.testing.assert_allclose(baseline.update_templates(V, W, H),
                                   W * (Q @ H.T) / H.sum(axis=1)[None, :], rtol=1e-12)

    def test_exact_factorization_is_a_fixed_point(self, baseline, rng):
        W = rng.uniform(0.1, 1, (6, 2)); H = rng.uniform(0.1, 1, (2, 8)); V = W @ H
        np.testing.assert_allclose(baseline.update_activations(V, W, H), H, rtol=1e-12)
        np.testing.assert_allclose(baseline.update_templates(V, W, H), W, rtol=1e-12)

    @pytest.mark.parametrize("seed", range(5))
    def test_single_updates_do_not_increase_kl(self, baseline, seed):
        r = np.random.default_rng(seed)
        V = r.uniform(0, 2, (10, 15)); W = r.uniform(0.1, 1, (10, 3)); H = r.uniform(0.1, 1, (3, 15))
        d0 = baseline.kl_divergence(V, W @ H)
        H1 = baseline.update_activations(V, W, H)
        d1 = baseline.kl_divergence(V, W @ H1)
        W1 = baseline.update_templates(V, W, H1)
        d2 = baseline.kl_divergence(V, W1 @ H1)
        assert d1 <= d0 * (1 + 1e-12) and d2 <= d1 * (1 + 1e-12)

    def test_return_new_arrays_and_do_not_mutate(self, baseline, rng):
        V = rng.uniform(0.1, 1, (5, 6)); W = rng.uniform(0.1, 1, (5, 2)); H = rng.uniform(0.1, 1, (2, 6))
        V0, W0, H0 = V.copy(), W.copy(), H.copy()
        H1 = baseline.update_activations(V, W, H)
        W1 = baseline.update_templates(V, W, H)
        assert H1 is not H and W1 is not W
        assert not np.shares_memory(H1, H) and not np.shares_memory(W1, W)
        np.testing.assert_array_equal(V, V0); np.testing.assert_array_equal(W, W0); np.testing.assert_array_equal(H, H0)

    def test_nonnegative_shapes_and_finite_with_zeros(self, baseline, rng):
        V = rng.uniform(0, 1, (6, 7)); V[:, 2] = 0.0; V[3, :] = 0.0
        W = rng.uniform(0.1, 1, (6, 3)); W[:, 1] = 0.0          # a dead template
        H = rng.uniform(0.1, 1, (3, 7)); H[2, :] = 0.0          # a dead activation row
        H1 = baseline.update_activations(V, W, H)
        W1 = baseline.update_templates(V, W, H)
        for X, shape in ((H1, (3, 7)), (W1, (6, 3))):
            assert X.shape == shape and np.isfinite(X).all() and (X >= 0).all()


# ----------------------------------------------------------------------------
# normalize_templates (Eq. 8)
# ----------------------------------------------------------------------------
class TestNormalize:
    def test_hand_example(self, baseline):
        out = baseline.normalize_templates(np.array([[0.6], [1.4]]), np.array([[2.0, 3.0]]))
        assert set(out) == {"W", "H"}
        np.testing.assert_allclose(out["W"], [[0.3], [0.7]], atol=1e-12)
        np.testing.assert_allclose(out["H"], [[4.0, 6.0]], atol=1e-12)

    def test_product_unchanged_and_columns_sum_to_one(self, baseline, rng):
        W = rng.uniform(0.1, 3, (8, 4)); H = rng.uniform(0.1, 3, (4, 10))
        out = baseline.normalize_templates(W, H)
        np.testing.assert_allclose(out["W"] @ out["H"], W @ H, rtol=1e-12)
        np.testing.assert_allclose(out["W"].sum(axis=0), 1.0, rtol=1e-12)

    def test_zero_column_is_safe(self, baseline):
        W = np.array([[0.0, 1.0], [0.0, 3.0]]); H = np.ones((2, 3))
        out = baseline.normalize_templates(W, H)
        assert np.isfinite(out["W"]).all() and np.isfinite(out["H"]).all()
        np.testing.assert_allclose(out["W"] @ out["H"], W @ H, atol=1e-12)

    def test_inputs_not_mutated(self, baseline, rng):
        W = rng.uniform(0.1, 3, (5, 2)); H = rng.uniform(0.1, 3, (2, 4)); W0, H0 = W.copy(), H.copy()
        baseline.normalize_templates(W, H)
        np.testing.assert_array_equal(W, W0); np.testing.assert_array_equal(H, H0)


# ----------------------------------------------------------------------------
# fit_nmf (joint factorization)
# ----------------------------------------------------------------------------
class TestFitNMF:
    def test_output_contract(self, baseline, positive_V):
        run = baseline.fit_nmf(positive_V, 3, n_iter=25, seed=4)
        assert set(run) == FIT_KEYS, f"keys must be exactly {sorted(FIT_KEYS)}"
        assert run["W"].shape == (12, 3) and run["H"].shape == (3, 20)
        assert run["loss"].shape == (26,)
        assert np.issubdtype(np.asarray(run["loss"]).dtype, np.floating)
        assert is_integer(run["seed"]) and run["seed"] == 4
        assert is_integer(run["n_iter"]) and run["n_iter"] == 25

    def test_equivalent_to_spec_loop(self, baseline, positive_V):
        """init_factors -> (update H, update W, normalize) x n_iter, nothing else."""
        n_iter, seed = 7, 3
        f = baseline.init_factors(positive_V, 3, seed); W, H = f["W"], f["H"]
        for _ in range(n_iter):
            H = baseline.update_activations(positive_V, W, H)
            W = baseline.update_templates(positive_V, W, H)
            n = baseline.normalize_templates(W, H); W, H = n["W"], n["H"]
        run = baseline.fit_nmf(positive_V, 3, n_iter=n_iter, seed=seed)
        np.testing.assert_allclose(run["W"], W, rtol=1e-10)
        np.testing.assert_allclose(run["H"], H, rtol=1e-10)

    def test_loss_starts_at_initial_loss(self, baseline, positive_V):
        W, H = spec_init(positive_V, 3, 2)
        run = baseline.fit_nmf(positive_V, 3, n_iter=5, seed=2)
        assert run["loss"][0] == pytest.approx(baseline.kl_divergence(positive_V, W @ H), rel=1e-10)
        assert run["loss"][-1] == pytest.approx(
            baseline.kl_divergence(positive_V, run["W"] @ run["H"]), rel=1e-10)

    def test_monotone_on_random_data(self, baseline, positive_V):
        assert nonincreasing(baseline.fit_nmf(positive_V, 4, n_iter=100, seed=0)["loss"])

    def test_monotone_on_clip(self, baseline, clip0):
        run = baseline.fit_nmf(clip0["V"], 3, n_iter=50, seed=0)
        assert nonincreasing(run["loss"]), "KL increased during fit_nmf on make_clip(0)"
        assert run["loss"][-1] < run["loss"][0]

    def test_factors_valid(self, baseline, clip0):
        run = baseline.fit_nmf(clip0["V"], 3, n_iter=20, seed=1)
        assert run["W"].shape == (F_BINS, 3) and run["H"].shape == (3, T_FRAMES)
        for X in (run["W"], run["H"]):
            assert np.isfinite(X).all() and (X >= 0).all()
        np.testing.assert_allclose(run["W"].sum(axis=0), 1.0, rtol=1e-10)

    def test_deterministic(self, baseline, positive_V):
        a = baseline.fit_nmf(positive_V, 3, n_iter=10, seed=9)
        b = baseline.fit_nmf(positive_V, 3, n_iter=10, seed=9)
        np.testing.assert_array_equal(a["W"], b["W"]); np.testing.assert_array_equal(a["loss"], b["loss"])

    def test_track_loss_false(self, baseline, positive_V):
        a = baseline.fit_nmf(positive_V, 3, n_iter=10, seed=1, track_loss=False)
        b = baseline.fit_nmf(positive_V, 3, n_iter=10, seed=1, track_loss=True)
        assert np.asarray(a["loss"]).size == 0
        np.testing.assert_allclose(a["W"], b["W"], rtol=1e-12)
        np.testing.assert_allclose(a["H"], b["H"], rtol=1e-12)

    def test_zero_iterations(self, baseline, positive_V):
        run = baseline.fit_nmf(positive_V, 2, n_iter=0, seed=0)
        assert np.asarray(run["loss"]).shape == (1,)

    @pytest.mark.parametrize("seed", range(5))
    def test_rank_one_closed_form(self, baseline, seed):
        """Appendix C: for K = 1 one joint iteration gives WH = r c^T / S from any start."""
        V = np.random.default_rng(100 + seed).uniform(0.1, 3, (5, 7))
        run = baseline.fit_nmf(V, 1, n_iter=1, seed=seed)
        expected = np.outer(V.sum(axis=1), V.sum(axis=0)) / V.sum()
        np.testing.assert_allclose(run["W"] @ run["H"], expected, rtol=1e-10)

    def test_hand_example_optimum(self, baseline, hand_V):
        run = baseline.fit_nmf(hand_V, 1, n_iter=3, seed=0)
        np.testing.assert_allclose(run["W"] @ run["H"], [[1.2, 1.8], [2.8, 4.2]], rtol=1e-10)
        assert run["loss"][-1] == pytest.approx(0.0402174, abs=1e-6)

    def test_exact_low_rank_recovery(self, baseline):
        rng = np.random.default_rng(7)
        V = rng.uniform(0, 1, (6, 2)) @ rng.uniform(0, 1, (2, 8))
        loss = baseline.fit_nmf(V, 2, n_iter=500, seed=0)["loss"]
        assert loss[0] == pytest.approx(4.868, rel=2e-3)
        assert loss[10] == pytest.approx(0.508, rel=2e-3)
        assert loss[100] < 1e-10
        assert loss[-1] < 1e-8
        assert nonincreasing(loss[:100], rtol=1e-7)

    def test_zero_frame_and_zero_bin(self, baseline, positive_V):
        V = positive_V.copy(); V[:, 4] = 0.0; V[6, :] = 0.0
        run = baseline.fit_nmf(V, 3, n_iter=30, seed=0)
        Z = run["W"] @ run["H"]
        assert np.isfinite(run["W"]).all() and np.isfinite(run["H"]).all()
        np.testing.assert_allclose(Z[:, 4], 0.0, atol=1e-12)
        np.testing.assert_allclose(Z[6, :], 0.0, atol=1e-12)

    def test_integer_input_accepted(self, baseline):
        V = np.arange(1, 13).reshape(3, 4)
        run = baseline.fit_nmf(V, 2, n_iter=5, seed=0)
        assert np.isfinite(run["W"]).all()

    def test_input_not_mutated(self, baseline, positive_V):
        V = positive_V.copy()
        baseline.fit_nmf(V, 3, n_iter=5, seed=0)
        np.testing.assert_array_equal(V, positive_V)

    @pytest.mark.parametrize("bad_V", [
        np.array([[1.0, -1.0], [1.0, 1.0]]),
        np.array([[1.0, np.nan], [1.0, 1.0]]),
        np.array([[1.0, np.inf], [1.0, 1.0]]),
    ])
    def test_invalid_V_raises(self, baseline, bad_V):
        with pytest.raises(ValueError):
            baseline.fit_nmf(bad_V, 1, n_iter=2)

    def test_invalid_K_and_iterations_raise(self, baseline, positive_V):
        with pytest.raises(ValueError):
            baseline.fit_nmf(positive_V, 0, n_iter=2)
        with pytest.raises(ValueError):
            baseline.fit_nmf(positive_V, 2, n_iter=-1)


# ----------------------------------------------------------------------------
# fit_nmf_restarts
# ----------------------------------------------------------------------------
class TestRestarts:
    def test_selects_lowest_final_loss(self, baseline, positive_V):
        seeds = [3, 0, 7, 1]
        out = baseline.fit_nmf_restarts(positive_V, 3, seeds=seeds, n_iter=15)
        assert FIT_KEYS | {"best_seed", "final_losses"} <= set(out)
        singles = [float(baseline.fit_nmf(positive_V, 3, n_iter=15, seed=s)["loss"][-1]) for s in seeds]
        assert len(out["final_losses"]) == len(seeds)
        np.testing.assert_allclose(out["final_losses"], singles, rtol=1e-10, err_msg="final_losses must be in input seed order")
        assert all(isinstance(x, float) for x in out["final_losses"])
        assert out["best_seed"] == seeds[int(np.argmin(singles))]
        best = baseline.fit_nmf(positive_V, 3, n_iter=15, seed=out["best_seed"])
        np.testing.assert_allclose(out["W"], best["W"], rtol=1e-10)
        np.testing.assert_allclose(out["H"], best["H"], rtol=1e-10)
        assert out["seed"] == out["best_seed"]

    def test_single_seed(self, baseline, positive_V):
        out = baseline.fit_nmf_restarts(positive_V, 2, seeds=[5], n_iter=5)
        assert out["best_seed"] == 5 and len(out["final_losses"]) == 1


# ----------------------------------------------------------------------------
# infer_activations (frozen dictionary)
# ----------------------------------------------------------------------------
class TestInferActivations:
    def test_output_contract(self, baseline, positive_V, rng):
        W = rng.uniform(0.1, 1, (12, 3))
        out = baseline.infer_activations(positive_V, W, n_iter=10, seed=0)
        assert set(out) == {"H", "loss"}
        assert out["H"].shape == (3, 20)
        assert np.asarray(out["loss"]).shape == (11,)
        assert np.isfinite(out["H"]).all() and (out["H"] >= 0).all()

    def test_equivalent_to_spec_loop(self, baseline, positive_V, rng):
        """Spec init, then only H updates; W is not updated and not renormalized."""
        W = 3.0 * rng.uniform(0.1, 1, (12, 3))           # columns deliberately do not sum to 1
        H = spec_infer_init(positive_V, W, seed=11)
        for _ in range(6):
            H = baseline.update_activations(positive_V, W, H)
        out = baseline.infer_activations(positive_V, W, n_iter=6, seed=11)
        np.testing.assert_allclose(out["H"], H, rtol=1e-10, err_msg=(
            "Initialize H0 ~ U(0.5,1.5) from default_rng(seed), scale by mean(V)/mean(W H0), "
            "then apply update_activations only."))

    def test_W_not_mutated(self, baseline, positive_V, rng):
        W = rng.uniform(0.1, 1, (12, 3)); W0 = W.copy()
        baseline.infer_activations(positive_V, W, n_iter=5)
        np.testing.assert_array_equal(W, W0)

    def test_loss_monotone_and_starts_at_init(self, baseline, positive_V, rng):
        W = rng.uniform(0.1, 1, (12, 3))
        out = baseline.infer_activations(positive_V, W, n_iter=50, seed=2)
        H0 = spec_infer_init(positive_V, W, seed=2)
        assert out["loss"][0] == pytest.approx(baseline.kl_divergence(positive_V, W @ H0), rel=1e-10)
        assert nonincreasing(out["loss"])

    def test_recovers_exact_activations(self, baseline, rng):
        W = rng.uniform(0.1, 1, (20, 3)); H_true = rng.uniform(0.1, 1, (3, 15)); V = W @ H_true
        out = baseline.infer_activations(V, W, n_iter=3000, seed=0)
        np.testing.assert_allclose(W @ out["H"], V, rtol=1e-4)

    def test_track_loss_false(self, baseline, positive_V, rng):
        W = rng.uniform(0.1, 1, (12, 3))
        a = baseline.infer_activations(positive_V, W, n_iter=8, seed=1, track_loss=False)
        b = baseline.infer_activations(positive_V, W, n_iter=8, seed=1, track_loss=True)
        assert np.asarray(a["loss"]).size == 0
        np.testing.assert_allclose(a["H"], b["H"], rtol=1e-12)

    def test_shape_mismatch_raises(self, baseline, positive_V):
        with pytest.raises(ValueError):
            baseline.infer_activations(positive_V, np.ones((11, 3)), n_iter=2)
