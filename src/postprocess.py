"""Post-hoc smoothing of activations: the control for the smoothness penalty.

The modification (:mod:`src.modified`) smooths activations *inside* the
factorization, trading KL fit for smoothness at every update. This control
smooths *after* baseline inference instead: ``H`` is filtered along time,
frame by frame, before thresholding. If the control matches or beats the
penalty, the penalty's benefit (or harm) comes from smoothing itself rather
than from building it into the objective.

Two filters, each a centred window of odd width ``w`` frames, with the edges
handled by repeating the first and last frame (``mode="nearest"``):

- ``"mean"``: moving average, a linear low-pass filter, the closest analogue
  of the quadratic penalty;
- ``"median"``: running median, a common transcription post-process that keeps
  sharp onsets and removes runs shorter than about ``w / 2`` frames.
"""

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

FILTERS = ("mean", "median")


def smooth_activations(H, window, kind="mean"):
    """Filter each activation row along time with a centred window.

    Parameters
    ----------
    H : array_like, shape (K, T)
        Non-negative activations.
    window : int
        Odd window width in frames (>= 1). ``window == 1`` returns a copy of
        ``H``.
    kind : {"mean", "median"}, optional
        Moving average or running median.

    Returns
    -------
    ndarray, shape (K, T)
        Filtered activations, non-negative. The input is not modified. Rows
        are filtered independently; a row that is constant in time is
        unchanged.

    Raises
    ------
    ValueError
        If ``window`` is not a positive odd integer or ``kind`` is unknown.
    """
    if int(window) != window or window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer.")
    if kind not in FILTERS:
        raise ValueError(f"kind must be one of {FILTERS}.")
    H = np.array(H, dtype=float)  # copy
    window = int(window)
    if window == 1:
        return H
    if kind == "mean":
        out = uniform_filter1d(H, size=window, axis=1, mode="nearest")
    else:
        out = median_filter(H, size=(1, window), mode="nearest")
    return np.maximum(out, 0.0)  # guard against round-off below zero
