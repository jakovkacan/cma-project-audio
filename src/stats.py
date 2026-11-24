"""Uncertainty estimates for per-clip evaluation results.

The clip is the unit of analysis: frames within a clip are dependent, so
intervals are computed over one value per clip (e.g. F1, or the paired
difference ``d_i = x_i^mod - x_i^base``), never over individual frames.
"""

import numpy as np
import scipy


def mean_ci(values, level=0.95):
    """Compute the mean and a two-sided Student-t confidence interval.

    The interval is ``mean ± t_{1 - alpha/2, n-1} * s / sqrt(n)``, where
    ``alpha = 1 - level`` and ``s`` is the sample standard deviation with
    ``n - 1`` in the denominator (Eq. 16 of the project guide).

    Parameters
    ----------
    values : array_like of float, shape (n,)
        One value per clip, treated as independent samples.
    level : float, optional
        Confidence level in ``(0, 1)``. Default 0.95.

    Returns
    -------
    dict
        ``"mean"`` : float -- sample mean.
        ``"low"``, ``"high"`` : float -- lower and upper interval bounds.
        ``"sd"`` : float -- sample standard deviation (``ddof=1``).
        ``"n"`` : int -- number of values.

    Notes
    -----
    With fewer than two values the standard deviation is undefined, so
    ``"sd"``, ``"low"`` and ``"high"`` are NaN.
    """
    values = np.asarray(values)
    n = values.size

    mean = np.mean(values)
    sd = np.std(values, ddof=1)

    alpha = 1 - level
    q = scipy.stats.t.ppf(1 - alpha / 2, n - 1) * (sd / np.sqrt(n))

    return {
        "mean": mean,
        "low": mean - q,
        "high": mean + q,
        "sd": sd,
        "n": n
    }
