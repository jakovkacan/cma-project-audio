"""Synthetic harmonic audio data for the NMF experiments.

Generates short clips of three harmonic tones (A3, C#4, E4) with random
timing, turns them into magnitude spectrograms ``V`` of shape ``(F, T)`` with
frame-aligned note activity targets ``Y`` of shape ``(3, T)``, and provides
helpers for splitting seeds into dev/val/test sets and concatenating clips.
"""

import numpy as np
from scipy.signal import stft


def make_music(seed, fs=8000, seconds=6):
    """Synthesize a harmonic audio scene and return its spectrogram.

    Reference generator, copied unchanged from the project specification
    document. :func:`make_clip` is the same generator with a configurable noise
    level and a dict return value.

    Each of the three pitches (220.0, 277.18 and 329.63 Hz) plays twice: once at
    a fixed start time ``0.2 + 0.8 * k`` s and once at a random start in
    ``[3.0, 4.8)`` s. Each note lasts a random ``0.35``-``0.65`` s, has a
    20 ms attack and 0.5 s exponential decay, and is built from up to 6
    harmonics (amplitude ``1 / h``, random phase) below Nyquist. Gaussian noise
    with standard deviation 0.005 is added, and the signal is analyzed with a
    Hann-windowed STFT (``nperseg=512``, ``noverlap=384``, hop 128 samples).

    Parameters
    ----------
    seed : int or None or numpy.random.Generator
        Anything accepted by ``numpy.random.default_rng``.
    fs : int, optional
        Sampling frequency in Hz.
    seconds : float, optional
        Duration of the signal in seconds.

    Returns
    -------
    V : ndarray, shape (F, T)
        Magnitude spectrogram, with ``F = 257`` for ``nperseg=512``.
    Y : ndarray of bool, shape (3, T)
        Note activity, sampled at the center of each STFT frame. Row ``k``
        corresponds to the ``k``-th pitch.
    freq : ndarray, shape (F,)
        Frequency of each bin in Hz.
    frames : ndarray, shape (T,)
        Center time of each frame in seconds.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(int(fs * seconds)) / fs
    pitches = [220.0, 277.18, 329.63]
    audio = np.zeros_like(time)
    active = np.zeros((3, len(time)), dtype=bool)

    for k, f0 in enumerate(pitches):
        starts = [0.2 + 0.8 * k, rng.uniform(3.0, 4.8)]
        for start in starts:
            duration = rng.uniform(0.35, 0.65)
            age = time - start
            on = (age >= 0) & (age < duration)
            envelope = np.zeros_like(time)
            envelope[on] = (
                    (1 - np.exp(-age[on] / 0.02))
                    * np.exp(-age[on] / 0.5))
            tone = np.zeros_like(time)
            for harmonic in range(1, 7):
                frequency = harmonic * f0
                if frequency < fs / 2:
                    phase = rng.uniform(0, 2 * np.pi)
                    tone += np.sin(
                        2 * np.pi * frequency * time + phase
                    ) / harmonic
            audio += envelope * tone
            active[k] |= on

    audio += 0.005 * rng.normal(size=len(time))
    freq, frames, Z = stft(
        audio, fs=fs, window="hann", nperseg=512,
        noverlap=384, boundary=None, padded=False)
    indices = np.rint(frames * fs).astype(int)
    return np.abs(Z), active[:, indices], freq, frames


def make_clip(seed, noise_std=0.005, fs=8000, seconds=6):
    """Synthesize a harmonic audio clip with a configurable noise level.

    Same generator as :func:`make_music`, but the additive Gaussian noise level
    can be set, and the result is returned as a dict. For a given ``seed`` and
    ``noise_std=0.005`` the output matches :func:`make_music` exactly.

    Parameters
    ----------
    seed : int or None or numpy.random.Generator
        Anything accepted by ``numpy.random.default_rng``. Controls note
        timing, durations, harmonic phases and noise.
    noise_std : float, optional
        Standard deviation of the additive Gaussian noise. Must be finite and
        non-negative; ``0`` gives a noise-free clip.
    fs : int, optional
        Sampling frequency in Hz.
    seconds : float, optional
        Duration of the clip in seconds.

    Returns
    -------
    dict
        ``"V"`` : ndarray, shape (F, T) -- magnitude spectrogram.
        ``"Y"`` : ndarray of bool, shape (3, T) -- note activity per frame.
        ``"freq"`` : ndarray, shape (F,) -- bin frequencies in Hz.
        ``"times"`` : ndarray, shape (T,) -- frame center times in seconds.
        ``"seed"`` : the ``seed`` argument, unchanged.
        ``"noise_std"`` : float -- the ``noise_std`` argument.

    Raises
    ------
    ValueError
        If ``noise_std`` is negative or not finite.
    """
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("noise_std must be non-negative and finite.")

    rng = np.random.default_rng(seed)
    time = np.arange(int(fs * seconds)) / fs
    pitches = [220.0, 277.18, 329.63]
    audio = np.zeros_like(time)
    active = np.zeros((3, len(time)), dtype=bool)

    for k, f0 in enumerate(pitches):
        starts = [0.2 + 0.8 * k, rng.uniform(3.0, 4.8)]
        for start in starts:
            duration = rng.uniform(0.35, 0.65)
            age = time - start
            on = (age >= 0) & (age < duration)
            envelope = np.zeros_like(time)
            envelope[on] = (
                    (1 - np.exp(-age[on] / 0.02))
                    * np.exp(-age[on] / 0.5))
            tone = np.zeros_like(time)
            for harmonic in range(1, 7):
                frequency = harmonic * f0
                if frequency < fs / 2:
                    phase = rng.uniform(0, 2 * np.pi)
                    tone += np.sin(
                        2 * np.pi * frequency * time + phase
                    ) / harmonic
            audio += envelope * tone
            active[k] |= on

    audio += noise_std * rng.normal(size=len(time))
    freq, frames, Z = stft(
        audio, fs=fs, window="hann", nperseg=512,
        noverlap=384, boundary=None, padded=False)
    indices = np.rint(frames * fs).astype(int)

    return {
        "V": np.abs(Z),
        "Y": active[:, indices],
        "freq": freq,
        "times": frames,
        "seed": seed,
        "noise_std": noise_std
    }


def split_seeds(seed_config):
    """Expand seed intervals into dev/val/test seed lists.

    Each interval ``[start, stop]`` follows Python ``range`` semantics:
    ``start`` is included and ``stop`` is excluded.

    Parameters
    ----------
    seed_config : dict
        Must contain the keys ``"dev"``, ``"val"`` and ``"test"``, each mapped
        to a two-element ``[start, stop]`` list or tuple of ints. Other keys
        are ignored.

    Returns
    -------
    dict
        ``"dev"``, ``"val"``, ``"test"`` : list of int -- the seeds in each
        split.

    Raises
    ------
    ValueError
        If a required key is missing, an interval does not have exactly two
        values, an interval is empty, or two intervals overlap.

    Examples
    --------
    >>> split_seeds({"dev": [0, 3], "val": [100, 102], "test": [1000, 1002]})
    {'dev': [0, 1, 2], 'val': [100, 101], 'test': [1000, 1001]}
    """
    if not "dev" in seed_config or not "val" in seed_config or not "test" in seed_config:
        raise ValueError("seed_config must have values for keys 'dev', 'val', and 'test'.")

    dev_range = seed_config["dev"]
    val_range = seed_config["val"]
    test_range = seed_config["test"]

    if len(dev_range) != 2 or len(val_range) != 2 or len(test_range) != 2:
        raise ValueError("seed_config keys must be exactly [start, stop].")

    dev = np.asarray(list(range(dev_range[0], dev_range[1])))
    val = np.asarray(list(range(val_range[0], val_range[1])))
    test = np.asarray(list(range(test_range[0], test_range[1])))

    if dev.size == 0 or val.size == 0 or test.size == 0:
        raise ValueError("seed_config ranges must span at least 1 value.")

    intersect_dev_val = np.intersect1d(dev, val)
    intersect_val_test = np.intersect1d(val, test)
    intersect_dev_test = np.intersect1d(dev, test)
    if intersect_dev_val.size != 0 or intersect_dev_test.size != 0 or intersect_val_test.size != 0:
        raise ValueError("seed_config ranges produce overlaps.")

    return {
        "dev": list(range(dev_range[0], dev_range[1])),
        "val": list(range(val_range[0], val_range[1])),
        "test": list(range(test_range[0], test_range[1]))
    }


def concat_clips(clips):
    """Concatenate clips along the time axis.

    Parameters
    ----------
    clips : sequence of dict
        Clips as returned by :func:`make_clip`. Each must have ``"V"`` of
        shape ``(F, T_i)`` and ``"Y"`` of shape ``(3, T_i)``; ``F`` must be the
        same for every clip. Must be non-empty.

    Returns
    -------
    dict
        ``"V"`` : ndarray, shape (F, sum T_i) -- concatenated spectrograms.
        ``"Y"`` : ndarray, shape (3, sum T_i) -- concatenated activity targets.
        ``"bounds"`` : list of (int, int) -- ``(start, stop)`` frame indices of
        each clip in the result, half-open, so clip ``i`` is
        ``V[:, start:stop]``.

    Raises
    ------
    KeyError
        If a clip is missing ``"V"`` or ``"Y"``.
    ValueError
        If the clips have different ``F``, or a clip's ``V`` and ``Y`` have
        different numbers of frames.
    """
    V = [clip["V"] for clip in clips]
    Y = [clip["Y"] for clip in clips]
    bounds = []

    start = 0
    for clip in clips:
        length = clip["V"].shape[1]
        bounds.append((start, start + length))
        start = start + length

    shapesF = np.asarray([v.shape[0] for v in V])
    if not np.all(shapesF == shapesF[0]):
        raise ValueError("Clips disagree in shape of V")

    shapesVT = np.asarray([v.shape[1] for v in V])
    shapesYT = np.asarray([y.shape[1] for y in Y])
    if not np.all(shapesVT == shapesYT):
        raise ValueError("V and Y matrices must have equal T shapes.")
    V = np.concatenate(V, axis=1)
    Y = np.concatenate(Y, axis=1)

    return {
        "V": V,  # (F, sum T_i)
        "Y": Y,  # (3, sum T_i)
        "bounds": bounds  # (start, stop)
    }
