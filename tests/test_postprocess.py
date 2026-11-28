"""src/postprocess.py: post-hoc smoothing of activations (the control)."""
import numpy as np
import pytest


class TestSmoothActivations:
    @pytest.mark.parametrize("kind", ["mean", "median"])
    def test_window_one_is_identity_copy(self, postprocess, rng, kind):
        H = rng.uniform(0, 1, (3, 20))
        out = postprocess.smooth_activations(H, 1, kind)
        np.testing.assert_array_equal(out, H)
        assert out is not H

    def test_mean_hand_value(self, postprocess):
        # window 3, edges repeat the end frame: [1,1,4,1,1] -> [1, 2, 2, 2, 1]
        out = postprocess.smooth_activations(np.array([[1.0, 1.0, 4.0, 1.0, 1.0]]), 3, "mean")
        np.testing.assert_allclose(out, [[1.0, 2.0, 2.0, 2.0, 1.0]])

    def test_median_removes_single_frame_spike_and_keeps_step(self, postprocess):
        spike = postprocess.smooth_activations(np.array([[0.0, 0.0, 5.0, 0.0, 0.0]]), 3, "median")
        np.testing.assert_array_equal(spike, [[0.0, 0.0, 0.0, 0.0, 0.0]])
        step = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
        np.testing.assert_array_equal(postprocess.smooth_activations(step, 3, "median"), step)

    @pytest.mark.parametrize("kind", ["mean", "median"])
    def test_constant_rows_unchanged(self, postprocess, kind):
        H = np.repeat(np.array([[0.5], [2.0]]), 15, axis=1)
        np.testing.assert_allclose(postprocess.smooth_activations(H, 5, kind), H)

    @pytest.mark.parametrize("kind", ["mean", "median"])
    def test_rows_independent_shape_nonnegative(self, postprocess, rng, kind):
        H = rng.uniform(0, 1, (3, 30))
        out = postprocess.smooth_activations(H, 7, kind)
        assert out.shape == H.shape and (out >= 0).all() and np.isfinite(out).all()
        H2 = H.copy(); H2[1] = 0.0
        out2 = postprocess.smooth_activations(H2, 7, kind)
        np.testing.assert_array_equal(out2[[0, 2]], out[[0, 2]])

    def test_input_not_mutated(self, postprocess, rng):
        H = rng.uniform(0, 1, (3, 20)); H0 = H.copy()
        postprocess.smooth_activations(H, 5, "mean")
        postprocess.smooth_activations(H, 5, "median")
        np.testing.assert_array_equal(H, H0)

    @pytest.mark.parametrize("window", [0, 2, -3, 2.5])
    def test_bad_window_raises(self, postprocess, window):
        with pytest.raises(ValueError):
            postprocess.smooth_activations(np.ones((2, 5)), window)

    def test_bad_kind_raises(self, postprocess):
        with pytest.raises(ValueError):
            postprocess.smooth_activations(np.ones((2, 5)), 3, "gaussian")
