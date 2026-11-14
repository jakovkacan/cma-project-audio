"""Section 2 of the guide: src/data.py (make_music, make_clip, split_seeds, concat_clips)."""
import numpy as np
import pytest

from conftest import F_BINS, N_PITCHES, T_FRAMES, is_integer

CLIP_KEYS = {"V", "Y", "freq", "times", "seed", "noise_std"}


# ----------------------------------------------------------------------------
# make_music: the guide's generator, copied verbatim
# ----------------------------------------------------------------------------
class TestMakeMusic:
    def test_returns_four_arrays_with_guide_shapes(self, data):
        out = data.make_music(0)
        assert isinstance(out, tuple) and len(out) == 4, "make_music must return (V, Y, freq, frames)"
        V, Y, freq, frames = out
        assert V.shape == (F_BINS, T_FRAMES)
        assert Y.shape == (N_PITCHES, T_FRAMES)
        assert Y.dtype == bool
        assert freq.shape == (F_BINS,)
        assert frames.shape == (T_FRAMES,)
        assert (V >= 0).all() and np.isfinite(V).all()

    def test_deterministic_per_seed(self, data):
        a, b = data.make_music(3), data.make_music(3)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_different_seeds_differ(self, data):
        assert not np.allclose(data.make_music(0)[0], data.make_music(1)[0])


# ----------------------------------------------------------------------------
# make_clip: dictionary wrapper with a noise knob
# ----------------------------------------------------------------------------
class TestMakeClip:
    def test_exact_keys(self, clip0):
        assert set(clip0) == CLIP_KEYS

    def test_shapes_and_dtypes(self, clip0):
        assert clip0["V"].shape == (F_BINS, T_FRAMES)
        assert clip0["V"].dtype == np.float64
        assert clip0["Y"].shape == (N_PITCHES, T_FRAMES)
        assert clip0["Y"].dtype == bool
        assert clip0["freq"].shape == (F_BINS,)
        assert clip0["times"].shape == (T_FRAMES,)

    def test_metadata_types(self, clip0):
        assert is_integer(clip0["seed"]) and clip0["seed"] == 0
        assert isinstance(clip0["noise_std"], float) and clip0["noise_std"] == pytest.approx(0.005)

    def test_spectrogram_is_nonnegative_and_finite(self, clip0):
        assert (clip0["V"] >= 0).all()
        assert np.isfinite(clip0["V"]).all()

    def test_frequency_axis(self, clip0):
        # nu_f = f * fs / N = 15.625 f Hz, up to Nyquist (4000 Hz).
        np.testing.assert_allclose(clip0["freq"], 15.625 * np.arange(F_BINS))

    def test_frame_centre_times(self, clip0):
        # tau_t = (N/2 + t R) / fs  ->  0.032 s ... 5.968 s, step 0.016 s.
        np.testing.assert_allclose(clip0["times"], (256 + 128 * np.arange(T_FRAMES)) / 8000)

    def test_default_reproduces_make_music_exactly(self, data, clip0):
        V, Y, freq, frames = data.make_music(0)
        np.testing.assert_array_equal(clip0["V"], V)
        np.testing.assert_array_equal(clip0["Y"], Y)
        np.testing.assert_array_equal(clip0["freq"], freq)
        np.testing.assert_array_equal(clip0["times"], frames)

    def test_reference_values_seed0(self, clip0):
        np.testing.assert_array_equal(clip0["Y"].sum(axis=1), [59, 72, 69])
        assert clip0["V"].max() == pytest.approx(0.4175, rel=1e-3)
        assert clip0["V"].mean() == pytest.approx(3.335e-3, rel=1e-3)

    @pytest.mark.parametrize("seed", [0, 5, 1000])
    def test_noise_level_keeps_cases_paired(self, data, seed):
        quiet = data.make_clip(seed, noise_std=0.005)
        loud = data.make_clip(seed, noise_std=0.2)
        np.testing.assert_array_equal(quiet["Y"], loud["Y"], err_msg=(
            "Labels changed with noise_std: the noise must be the LAST random draw "
            "and only its coefficient may change."))
        assert not np.allclose(quiet["V"], loud["V"])
        assert loud["noise_std"] == pytest.approx(0.2)

    def test_more_noise_raises_the_floor(self, data):
        quiet = data.make_clip(7, noise_std=0.005)["V"]
        loud = data.make_clip(7, noise_std=0.2)["V"]
        assert np.median(loud) > np.median(quiet)

    def test_zero_noise_is_allowed(self, data):
        clip = data.make_clip(2, noise_std=0.0)
        assert np.isfinite(clip["V"]).all() and (clip["V"] >= 0).all()

    @pytest.mark.parametrize("bad", [-0.01, np.nan, np.inf])
    def test_invalid_noise_raises(self, data, bad):
        with pytest.raises(ValueError):
            data.make_clip(0, noise_std=bad)

    @pytest.mark.parametrize("seed", range(5))
    def test_every_pitch_has_isolated_frames(self, data, seed):
        # The first note of each pitch is always heard alone (Section 2.1).
        Y = data.make_clip(seed)["Y"]
        exclusive = Y.sum(axis=0) == 1
        for k in range(N_PITCHES):
            assert (Y[k] & exclusive).any(), f"pitch {k} has no isolated frames for seed {seed}"


# ----------------------------------------------------------------------------
# split_seeds
# ----------------------------------------------------------------------------
class TestSplitSeeds:
    CFG = {"dev": [0, 10], "val": [100, 110], "test": [1000, 1030]}

    def test_half_open_ranges(self, data):
        s = data.split_seeds(self.CFG)
        assert set(s) == {"dev", "val", "test"}
        assert s["dev"] == list(range(0, 10))
        assert s["val"] == list(range(100, 110))
        assert s["test"] == list(range(1000, 1030))

    def test_python_int_lists(self, data):
        s = data.split_seeds(self.CFG)
        for name, seeds in s.items():
            assert isinstance(seeds, list), f"{name} must be a list"
            assert all(type(x) is int for x in seeds), f"{name} must contain Python ints"

    def test_input_not_mutated(self, data):
        cfg = {k: list(v) for k, v in self.CFG.items()}
        data.split_seeds(cfg)
        assert cfg == self.CFG

    @pytest.mark.parametrize("cfg", [
        {"dev": [0, 10], "val": [5, 15], "test": [1000, 1030]},
        {"dev": [0, 10], "val": [100, 110], "test": [105, 120]},
        {"dev": [0, 10], "val": [100, 110], "test": [0, 3]},
    ])
    def test_overlap_raises(self, data, cfg):
        with pytest.raises(ValueError):
            data.split_seeds(cfg)

    @pytest.mark.parametrize("cfg", [
        {"dev": [0, 0], "val": [100, 110], "test": [1000, 1030]},
        {"dev": [0, 10], "val": [100, 100], "test": [1000, 1030]},
        {"dev": [0, 10], "val": [100, 110], "test": [1030, 1000]},
    ])
    def test_empty_raises(self, data, cfg):
        with pytest.raises(ValueError):
            data.split_seeds(cfg)


# ----------------------------------------------------------------------------
# concat_clips
# ----------------------------------------------------------------------------
class TestConcatClips:
    def test_shapes_and_bounds(self, data):
        clips = [data.make_clip(s) for s in (0, 1, 2)]
        out = data.concat_clips(clips)
        assert out["V"].shape == (F_BINS, 3 * T_FRAMES)
        assert out["Y"].shape == (N_PITCHES, 3 * T_FRAMES)
        bounds = [tuple(int(x) for x in b) for b in out["bounds"]]
        assert bounds == [(0, 372), (372, 744), (744, 1116)]

    def test_content_preserved_by_bounds(self, data):
        clips = [data.make_clip(s) for s in (4, 9)]
        out = data.concat_clips(clips)
        for clip, (a, b) in zip(clips, out["bounds"]):
            np.testing.assert_array_equal(out["V"][:, a:b], clip["V"])
            np.testing.assert_array_equal(out["Y"][:, a:b], clip["Y"])

    def test_development_set_shape(self, data):
        dev = data.concat_clips([data.make_clip(s) for s in range(10)])
        assert dev["V"].shape == (F_BINS, 3720)

    def test_inputs_not_mutated(self, data):
        clips = [data.make_clip(s) for s in (0, 1)]
        before = [c["V"].copy() for c in clips]
        data.concat_clips(clips)
        for c, b in zip(clips, before):
            np.testing.assert_array_equal(c["V"], b)

    def test_mismatched_F_raises(self, data):
        a = data.make_clip(0)
        b = data.make_clip(1)
        b = dict(b, V=b["V"][:-1])
        with pytest.raises(ValueError):
            data.concat_clips([a, b])
