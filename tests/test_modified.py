"""src/modified.py: KL-NMF inference with a temporal smoothness penalty on H."""
import numpy as np
import pytest

INFER_KEYS = {"H", "loss", "kl", "penalty", "mu"}


def nonincreasing(loss, rtol=1e-9):
    loss = np.asarray(loss)
    return bool(np.all(np.diff(loss) <= rtol * np.abs(loss[:-1]) + 1e-15))


def roughness(H):
    """Sum of squared frame-to-frame differences (the penalty without mu / 2)."""
    return float(np.sum(np.diff(H, axis=1) ** 2))


def loop_update(V, W, H, mu, eps=1e-10):
    """The modified update written element by element, as in the derivation."""
    K, T = H.shape
    Q = V / np.maximum(W @ H, eps)
    H_new = np.empty_like(H)
    for k in range(K):
        for t in range(T):
            N = (H[k, t - 1] if t > 0 else 0.0) + (H[k, t + 1] if t < T - 1 else 0.0)
            c = (t > 0) + (t < T - 1)
            num = W[:, k] @ Q[:, t] + mu * N
            den = W[:, k].sum() + mu * c * H[k, t]
            H_new[k, t] = H[k, t] * num / max(den, eps)
    return H_new


@pytest.fixture
def small_W(rng):
    return rng.uniform(0.1, 1, (12, 3))


# ----------------------------------------------------------------------------
# neighbour_sums and smoothness_penalty
# ----------------------------------------------------------------------------
class TestPenalty:
    def test_neighbour_sums_hand(self, modified):
        out = modified.neighbour_sums(np.array([[1.0, 2.0, 4.0]]))
        np.testing.assert_array_equal(out["N"], [[2.0, 5.0, 2.0]])
        np.testing.assert_array_equal(out["c"], [1.0, 2.0, 1.0])

    def test_neighbour_sums_single_frame(self, modified):
        out = modified.neighbour_sums(np.array([[3.0], [5.0]]))
        np.testing.assert_array_equal(out["N"], [[0.0], [0.0]])
        np.testing.assert_array_equal(out["c"], [0.0])

    def test_hand_value(self, modified):
        # rows: diffs (1, 2) and (-1, 0) -> squares sum to 1 + 4 + 1 + 0 = 6; mu/2 * 6 = 3 * 6 / 2
        H = np.array([[1.0, 2.0, 4.0], [3.0, 2.0, 2.0]])
        assert modified.smoothness_penalty(H, 3.0) == pytest.approx(9.0)

    def test_constant_in_time_is_zero(self, modified):
        H = np.repeat(np.array([[0.5], [2.0], [7.0]]), 10, axis=1)
        assert modified.smoothness_penalty(H, 100.0) == 0.0

    def test_mu_zero_and_single_frame_are_zero(self, modified, rng):
        assert modified.smoothness_penalty(rng.uniform(0, 1, (3, 9)), 0.0) == 0.0
        assert modified.smoothness_penalty(rng.uniform(0, 1, (3, 1)), 5.0) == 0.0

    def test_linear_in_mu(self, modified, rng):
        H = rng.uniform(0, 1, (3, 9))
        assert modified.smoothness_penalty(H, 4.0) == pytest.approx(4 * modified.smoothness_penalty(H, 1.0))

    @pytest.mark.parametrize("mu", [-1.0, np.nan, np.inf])
    def test_bad_mu_raises(self, modified, mu):
        with pytest.raises(ValueError):
            modified.smoothness_penalty(np.ones((2, 3)), mu)

    def test_objective_is_kl_plus_penalty(self, modified, baseline, positive_V, small_W, rng):
        H = rng.uniform(0.1, 1, (3, 20))
        obj = modified.smooth_objective(positive_V, small_W, H, 2.5)
        assert obj["kl"] == pytest.approx(baseline.kl_divergence(positive_V, small_W @ H), rel=1e-12)
        assert obj["penalty"] == pytest.approx(modified.smoothness_penalty(H, 2.5), rel=1e-12)
        assert obj["total"] == pytest.approx(obj["kl"] + obj["penalty"], rel=1e-12)


# ----------------------------------------------------------------------------
# update_activations_smooth
# ----------------------------------------------------------------------------
class TestUpdate:
    def test_hand_example(self, modified, hand_V):
        # W = [1, 1]^T, H = [1, 2], mu = 1.  WH = [[1, 2], [1, 2]], Q = [[1, 1], [3, 2]].
        # W^T Q = [4, 3], W^T 1 = 2, N = [2, 1], c = [1, 1].
        # H_new = [1 * (4 + 2) / (2 + 1), 2 * (3 + 1) / (2 + 2)] = [2, 2]
        # (the baseline update gives [2, 3]).
        W = np.ones((2, 1)); H = np.array([[1.0, 2.0]])
        np.testing.assert_allclose(modified.update_activations_smooth(hand_V, W, H, 1.0), [[2.0, 2.0]])

    @pytest.mark.parametrize("mu", [0.0, 0.3, 5.0])
    def test_matches_elementwise_derivation(self, modified, positive_V, small_W, rng, mu):
        H = rng.uniform(0.1, 1, (3, 20))
        np.testing.assert_allclose(modified.update_activations_smooth(positive_V, small_W, H, mu),
                                   loop_update(positive_V, small_W, H, mu), rtol=1e-12)

    def test_mu_zero_is_baseline_exactly(self, modified, baseline, positive_V, small_W, rng):
        H = rng.uniform(0.1, 1, (3, 20))
        np.testing.assert_array_equal(modified.update_activations_smooth(positive_V, small_W, H, 0.0),
                                      baseline.update_activations(positive_V, small_W, H))

    def test_nonnegative_finite_shape(self, modified, rng, small_W):
        V = rng.uniform(0, 1, (12, 20)); V[:, 4] = 0.0; V[2, :] = 0.0
        H = rng.uniform(0.1, 1, (3, 20))
        out = modified.update_activations_smooth(V, small_W, H, 10.0)
        assert out.shape == H.shape
        assert np.isfinite(out).all() and (out >= 0).all()

    def test_inputs_not_mutated(self, modified, positive_V, small_W, rng):
        H = rng.uniform(0.1, 1, (3, 20))
        copies = [positive_V.copy(), small_W.copy(), H.copy()]
        modified.update_activations_smooth(positive_V, small_W, H, 2.0)
        for a, b in zip([positive_V, small_W, H], copies):
            np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("mu", [0.1, 1.0, 10.0])
    def test_single_update_does_not_increase_objective(self, modified, positive_V, small_W, rng, mu):
        H = rng.uniform(0.1, 1, (3, 20))
        before = modified.smooth_objective(positive_V, small_W, H, mu)["total"]
        H1 = modified.update_activations_smooth(positive_V, small_W, H, mu)
        assert modified.smooth_objective(positive_V, small_W, H1, mu)["total"] <= before * (1 + 1e-12)


# ----------------------------------------------------------------------------
# infer_activations_smooth
# ----------------------------------------------------------------------------
class TestInferSmooth:
    def test_output_contract(self, modified, positive_V, small_W):
        out = modified.infer_activations_smooth(positive_V, small_W, 1.0, n_iter=10, seed=0)
        assert set(out) == INFER_KEYS
        assert out["H"].shape == (3, 20)
        for key in ("loss", "kl", "penalty"):
            assert np.asarray(out[key]).shape == (11,)
        assert np.isfinite(out["H"]).all() and (out["H"] >= 0).all()
        assert out["mu"] == 1.0

    def test_mu_zero_is_baseline_exactly(self, modified, baseline, positive_V, small_W):
        a = modified.infer_activations_smooth(positive_V, small_W, 0.0, n_iter=25, seed=7)
        b = baseline.infer_activations(positive_V, small_W, n_iter=25, seed=7)
        np.testing.assert_array_equal(a["H"], b["H"])
        np.testing.assert_allclose(a["loss"], b["loss"], rtol=1e-12)
        assert np.all(a["penalty"] == 0.0)

    def test_same_start_as_baseline(self, modified, baseline, positive_V, small_W):
        """Identical positive starts: the initial KL terms agree for any mu."""
        a = modified.infer_activations_smooth(positive_V, small_W, 5.0, n_iter=1, seed=3)
        b = baseline.infer_activations(positive_V, small_W, n_iter=1, seed=3)
        assert a["kl"][0] == pytest.approx(b["loss"][0], rel=1e-12)

    @pytest.mark.parametrize("mu", [0.01, 1.0, 100.0])
    @pytest.mark.parametrize("seed", range(3))
    def test_objective_monotone_on_random_data(self, modified, positive_V, small_W, mu, seed):
        out = modified.infer_activations_smooth(positive_V, small_W, mu, n_iter=100, seed=seed)
        assert nonincreasing(out["loss"])

    @pytest.mark.parametrize("mu", [1.0, 10.0, 100.0])
    def test_objective_monotone_on_clip(self, modified, baseline, clip0, mu):
        W = baseline.fit_nmf(clip0["V"], 3, n_iter=60, seed=0, track_loss=False)["W"]
        out = modified.infer_activations_smooth(clip0["V"], W, mu, n_iter=60, seed=0)
        assert nonincreasing(out["loss"])

    def test_larger_mu_gives_smoother_activations(self, modified, positive_V, small_W):
        """At (near) convergence, raising mu trades KL fit for smoothness."""
        outs = [modified.infer_activations_smooth(positive_V, small_W, mu, n_iter=2000, seed=0)
                for mu in (0.0, 1.0, 10.0)]
        rough = [roughness(o["H"]) for o in outs]
        kl = [o["kl"][-1] for o in outs]
        assert rough[0] > rough[1] > rough[2]
        assert kl[0] < kl[1] < kl[2]

    def test_constant_activations_recovered(self, modified, rng):
        """If the exact activations are constant in time, the penalty costs nothing."""
        W = rng.uniform(0.1, 1, (15, 3))
        H_true = np.repeat(rng.uniform(0.2, 1, (3, 1)), 12, axis=1)
        V = W @ H_true
        out = modified.infer_activations_smooth(V, W, 10.0, n_iter=3000, seed=0)
        np.testing.assert_allclose(W @ out["H"], V, rtol=1e-4)
        assert out["penalty"][-1] < 1e-8

    def test_single_frame_equals_baseline(self, modified, baseline, rng):
        W = rng.uniform(0.1, 1, (12, 3)); V = rng.uniform(0.1, 1, (12, 1))
        a = modified.infer_activations_smooth(V, W, 50.0, n_iter=20, seed=0)
        b = baseline.infer_activations(V, W, n_iter=20, seed=0)
        np.testing.assert_allclose(a["H"], b["H"], rtol=1e-12)

    def test_track_loss_false(self, modified, positive_V, small_W):
        a = modified.infer_activations_smooth(positive_V, small_W, 2.0, n_iter=8, seed=1, track_loss=False)
        b = modified.infer_activations_smooth(positive_V, small_W, 2.0, n_iter=8, seed=1, track_loss=True)
        for key in ("loss", "kl", "penalty"):
            assert np.asarray(a[key]).size == 0
        np.testing.assert_array_equal(a["H"], b["H"])

    def test_inputs_not_mutated(self, modified, positive_V, small_W):
        V0, W0 = positive_V.copy(), small_W.copy()
        modified.infer_activations_smooth(positive_V, small_W, 3.0, n_iter=5)
        np.testing.assert_array_equal(positive_V, V0)
        np.testing.assert_array_equal(small_W, W0)

    @pytest.mark.parametrize("kwargs", [
        {"mu": -1.0}, {"mu": np.nan}, {"mu": 1.0, "n_iter": -1},
    ])
    def test_bad_arguments_raise(self, modified, positive_V, small_W, kwargs):
        with pytest.raises(ValueError):
            modified.infer_activations_smooth(positive_V, small_W, **kwargs)

    def test_bad_V_raises(self, modified, small_W):
        V = np.ones((12, 4)); V[0, 0] = -1.0
        with pytest.raises(ValueError):
            modified.infer_activations_smooth(V, small_W, 1.0, n_iter=2)
        V[0, 0] = np.nan
        with pytest.raises(ValueError):
            modified.infer_activations_smooth(V, small_W, 1.0, n_iter=2)

    def test_shape_mismatch_raises(self, modified, positive_V):
        with pytest.raises(ValueError):
            modified.infer_activations_smooth(positive_V, np.ones((11, 3)), 1.0, n_iter=2)
