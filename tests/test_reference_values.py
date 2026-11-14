"""Reference values quoted in the guide (Sections 3.5, 4 and 6.1).

These reproduce the full development-set experiment, so they are marked ``slow``
(about one minute). They pass only if you follow the initialization and seed
conventions exactly. Small floating-point differences are covered by the tolerances.
"""
import json

import numpy as np
import pytest

from conftest import REPO_ROOT, unpack_evaluation

pytestmark = pytest.mark.slow


def _baseline_config(noise_levels=(0.005,)):
    path = REPO_ROOT / "configs" / "baseline.json"
    if path.exists():
        cfg = json.loads(path.read_text())
    else:
        from conftest import tiny_config
        cfg = tiny_config()
        cfg.update(seeds={"dev": [0, 10], "val": [100, 110], "test": [1000, 1030]},
                   restart_seeds=[0, 1, 2, 3], n_iter_dict=300, n_iter_infer=200)
    cfg["noise_levels"] = list(noise_levels)
    cfg["timing"] = {"warmup": 0, "repeats": 1}
    return cfg


# ----------------------------------------------------------------------------
# Section 3.5: blind joint factorization of make_clip(0)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("seed, d_per_entry, peaks", [
    (0, 6.69e-4, [250.0, 281.25, 328.125]),
    (1, 6.69e-4, [250.0, 281.25, 328.125]),
    (2, 3.69e-4, [218.75, 281.25, 328.125]),
    (3, 6.69e-4, [250.0, 281.25, 328.125]),
])
def test_blind_factorization_of_clip0(baseline, clip0, seed, d_per_entry, peaks):
    run = baseline.fit_nmf(clip0["V"], 3, n_iter=200, seed=seed)
    V = clip0["V"]
    assert run["loss"][-1] / V.size == pytest.approx(d_per_entry, rel=1e-2)
    found = np.sort(clip0["freq"][np.argmax(run["W"], axis=0)])
    np.testing.assert_allclose(found, peaks)


# ----------------------------------------------------------------------------
# Section 4: development dictionary (via learn_dictionary)
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dev_dictionary(protocol):
    return protocol.learn_dictionary(_baseline_config())


def test_restart_losses(dev_dictionary):
    per_entry = np.asarray(dev_dictionary["final_losses"]) / (257 * 3720)
    np.testing.assert_allclose(per_entry, [8.25e-4, 3.764e-4, 3.764e-4, 3.764e-4], rtol=5e-3)
    # Seeds 1-3 agree to about five significant figures, so any of them may win.
    assert dev_dictionary["best_seed"] in (1, 2, 3)


def test_exclusive_frame_counts(alignment, data):
    dev = data.concat_clips([data.make_clip(s) for s in range(10)])
    np.testing.assert_array_equal(alignment.reference_templates(dev["V"], dev["Y"])["n_frames"], [427, 486, 438])


def test_matched_similarities(dev_dictionary):
    np.testing.assert_allclose(dev_dictionary["match"]["matched"], [0.9998, 0.9998, 0.9991], atol=5e-4)


def test_aligned_template_peaks(dev_dictionary, clip0):
    peaks = clip0["freq"][np.argmax(np.asarray(dev_dictionary["W"]), axis=0)]
    np.testing.assert_allclose(peaks, [218.75, 281.25, 328.125])


def test_activation_scales(dev_dictionary):
    np.testing.assert_allclose(dev_dictionary["scales"], [2.14, 2.18, 2.17], atol=0.015)


# ----------------------------------------------------------------------------
# Section 6.1: expected baseline values at the default noise level
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def default_noise_evaluation(protocol, dev_dictionary):
    return unpack_evaluation(protocol.evaluate_setting(dev_dictionary, 0.005, _baseline_config()))


def _values(records, metric):
    return np.array([float(r["value"]) for r in records if r["metric"] == metric])


def test_selected_threshold(default_noise_evaluation):
    assert default_noise_evaluation[1] == pytest.approx(0.35)


def test_test_f1(default_noise_evaluation):
    f1 = _values(default_noise_evaluation[0], "f1")
    assert f1.size == 30
    assert f1.mean() == pytest.approx(0.993, abs=0.002)
    assert f1.std(ddof=1) == pytest.approx(0.004, abs=0.0015)
    assert f1.min() >= 0.98


def test_test_kl(default_noise_evaluation):
    assert _values(default_noise_evaluation[0], "kl_per_entry").mean() == pytest.approx(3.69e-4, rel=2e-2)


def test_no_missed_onsets(default_noise_evaluation):
    assert _values(default_noise_evaluation[0], "onsets_missed").sum() == 0
