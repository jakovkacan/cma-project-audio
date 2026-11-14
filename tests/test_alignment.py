"""Section 4 of the guide: src/alignment.py (reference templates, matching, scales)."""
import numpy as np
import pytest

from conftest import is_integer


def _toy():
    """F=4 bins, T=6 frames. Frames 0,1 only pitch 0; frame 2 pitches 0+1 (ignored);
    frame 3 only pitch 1; frame 4 only pitch 2; frame 5 silent."""
    V = np.array([
        [1.0, 3.0, 9.0, 0.0, 1.0, 5.0],
        [1.0, 1.0, 9.0, 2.0, 1.0, 5.0],
        [0.0, 0.0, 9.0, 2.0, 1.0, 5.0],
        [2.0, 0.0, 9.0, 0.0, 1.0, 5.0],
    ])
    Y = np.zeros((3, 6), dtype=bool)
    Y[0, [0, 1, 2]] = True
    Y[1, [2, 3]] = True
    Y[2, 4] = True
    return V, Y


class TestReferenceTemplates:
    def test_exclusive_frame_means(self, alignment):
        V, Y = _toy()
        out = alignment.reference_templates(V, Y)
        assert set(out) >= {"R", "n_frames"}
        R = out["R"]
        assert R.shape == (4, 3)
        np.testing.assert_allclose(R[:, 0], np.array([2.0, 1.0, 0.0, 1.0]) / 4.0)
        np.testing.assert_allclose(R[:, 1], np.array([0.0, 1.0, 1.0, 0.0]) / 2.0)
        np.testing.assert_allclose(R[:, 2], np.full(4, 0.25))
        np.testing.assert_array_equal(out["n_frames"], [2, 1, 1])
        assert np.issubdtype(np.asarray(out["n_frames"]).dtype, np.integer)

    def test_columns_sum_to_one(self, alignment, rng):
        V = rng.uniform(0, 1, (10, 30)); Y = np.zeros((3, 30), dtype=bool)
        Y[0, :10] = True; Y[1, 10:20] = True; Y[2, 20:] = True
        np.testing.assert_allclose(alignment.reference_templates(V, Y)["R"].sum(axis=0), 1.0)

    def test_pitch_without_exclusive_frames_raises(self, alignment):
        V, Y = _toy()
        Y[2, :] = False; Y[2, 2] = True          # pitch 2 only ever sounds with others
        with pytest.raises(ValueError):
            alignment.reference_templates(V, Y)

    def test_inputs_not_mutated(self, alignment):
        V, Y = _toy(); V0, Y0 = V.copy(), Y.copy()
        alignment.reference_templates(V, Y)
        np.testing.assert_array_equal(V, V0); np.testing.assert_array_equal(Y, Y0)


class TestCosineSimilarity:
    def test_shape_and_identity(self, alignment):
        S = alignment.cosine_similarity(np.eye(4)[:, :3], np.eye(4)[:, :2])
        assert S.shape == (3, 2)
        np.testing.assert_allclose(S, np.eye(4)[:3, :2])

    def test_matches_definition(self, alignment, rng):
        A = rng.uniform(0, 1, (8, 4)); B = rng.uniform(0, 1, (8, 3))
        expected = (A.T @ B) / np.outer(np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0))
        np.testing.assert_allclose(alignment.cosine_similarity(A, B), expected, rtol=1e-12)

    def test_scale_invariant_and_bounded(self, alignment, rng):
        A = rng.uniform(0, 1, (8, 4)); B = rng.uniform(0, 1, (8, 3))
        S = alignment.cosine_similarity(A, B)
        np.testing.assert_allclose(alignment.cosine_similarity(7 * A, 0.1 * B), S, rtol=1e-12)
        assert (S <= 1 + 1e-12).all() and (S >= -1 - 1e-12).all()

    def test_zero_column_is_finite(self, alignment):
        A = np.array([[0.0, 1.0], [0.0, 1.0]])
        S = alignment.cosine_similarity(A, np.ones((2, 1)))
        assert np.isfinite(S).all()
        assert S[0, 0] == pytest.approx(0.0, abs=1e-9)


class TestMatchComponents:
    @staticmethod
    def _references(rng, F=40):
        R = rng.uniform(0, 0.05, (F, 3))
        R[5, 0] += 1.0; R[12, 1] += 1.0; R[25, 2] += 1.0     # distinct peaks per pitch
        return R / R.sum(axis=0)

    def test_restores_a_cyclic_permutation(self, alignment, rng):
        R = self._references(rng)
        order = [1, 2, 0]                                     # not its own inverse
        W = R[:, order] * np.array([2.0, 3.0, 4.0]) + rng.uniform(0, 1e-3, (40, 3))
        out = alignment.match_components(W, R)
        assert set(out) >= {"perm", "similarity", "matched", "unmatched"}
        perm = np.asarray(out["perm"])
        assert perm.shape == (3,) and np.issubdtype(perm.dtype, np.integer)
        np.testing.assert_array_equal(perm, [2, 0, 1], err_msg="W[:, perm] must put pitch k's component in column k")
        aligned = alignment.cosine_similarity(W[:, perm], R)
        assert np.all(np.diag(aligned) > 0.99)
        assert list(out["unmatched"]) == []

    def test_similarity_and_matched_consistent(self, alignment, rng):
        R = self._references(rng)
        W = rng.uniform(0, 1, (40, 3))
        out = alignment.match_components(W, R)
        S = np.asarray(out["similarity"])
        assert S.shape == (3, 3)
        np.testing.assert_allclose(S, alignment.cosine_similarity(W, R), rtol=1e-12)
        np.testing.assert_allclose(out["matched"], S[np.asarray(out["perm"]), np.arange(3)], rtol=1e-12)

    def test_assignment_is_optimal(self, alignment, rng):
        import itertools
        R = self._references(rng)
        W = rng.uniform(0, 1, (40, 3))
        out = alignment.match_components(W, R)
        S = alignment.cosine_similarity(W, R)
        best = max(sum(S[p[k], k] for k in range(3)) for p in itertools.permutations(range(3)))
        assert float(np.sum(out["matched"])) == pytest.approx(best, rel=1e-12)

    def test_extra_component_is_unmatched(self, alignment, rng):
        R = self._references(rng)
        flat = np.full((40, 1), 1 / 40)                      # a wideband non-note component
        W = np.hstack([R[:, [2]], flat, R[:, [0]], R[:, [1]]])
        out = alignment.match_components(W, R)
        np.testing.assert_array_equal(out["perm"], [2, 3, 0])
        assert [int(x) for x in out["unmatched"]] == [1]
        assert np.asarray(out["similarity"]).shape == (4, 3)


class TestActivationScales:
    def test_percentile_of_active_frames_only(self, alignment):
        H = np.zeros((2, 20))
        H[0, :10] = np.arange(1, 11); H[0, 10:] = 1000.0     # huge values on inactive frames
        H[1, 5:15] = np.arange(10, 110, 10)
        Y = np.zeros((2, 20), dtype=bool); Y[0, :10] = True; Y[1, 5:15] = True
        c = np.asarray(alignment.activation_scales(H, Y))
        assert c.shape == (2,)
        np.testing.assert_allclose(c, [np.percentile(np.arange(1, 11), 95), np.percentile(np.arange(10, 110, 10), 95)])

    def test_q_parameter(self, alignment):
        H = np.arange(1, 11, dtype=float)[None, :]
        Y = np.ones((1, 10), dtype=bool)
        assert float(np.asarray(alignment.activation_scales(H, Y, q=50.0))[0]) == pytest.approx(5.5)

    def test_floor(self, alignment):
        c = np.asarray(alignment.activation_scales(np.zeros((3, 5)), np.ones((3, 5), dtype=bool)))
        np.testing.assert_allclose(c, 1e-12)
        assert (c > 0).all()

    def test_inputs_not_mutated(self, alignment, rng):
        H = rng.uniform(0, 1, (3, 8)); Y = rng.uniform(size=(3, 8)) > 0.5; H0 = H.copy()
        alignment.activation_scales(H, Y)
        np.testing.assert_array_equal(H, H0)
