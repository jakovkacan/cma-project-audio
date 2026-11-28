"""Modified NMF: KL divergence plus a temporal smoothness penalty on H.

The baseline (:mod:`src.baseline`) minimizes the generalized KL divergence
``D(V || WH)``. This module adds a quadratic penalty on frame-to-frame changes
of each activation row, following the temporal-continuity criterion of
Virtanen (2007):

.. math::

    J(W, H) = D(V \\| WH) + \\frac{\\mu}{2} \\sum_k \\sum_{t=2}^{T}
              (H_{kt} - H_{k,t-1})^2

with weight ``mu >= 0``. Only activation inference changes: the templates ``W``
are learned with the baseline and stay frozen. With ``mu = 0`` every function
here reproduces the baseline exactly.

Update rule
-----------
The penalty's gradient with respect to ``H_kt`` is

.. math:: \\mu (c_t H_{kt} - N_{kt}),

where ``N_kt = H_{k,t-1} + H_{k,t+1}`` sums the existing neighbours of frame
``t`` and ``c_t`` counts them (2 inside the clip, 1 at the first and last
frames, 0 if ``T == 1``). Splitting the full gradient of ``J`` into its
positive part ``W^T 1 + mu c H`` and negative part ``W^T Q + mu N`` (with
``Q = V / WH``) gives the multiplicative update

.. math::

    H_{kt} \\leftarrow H_{kt} \\,
        \\frac{\\sum_f W_{fk} Q_{ft} + \\mu N_{kt}}
             {\\sum_f W_{fk} + \\mu c_t H_{kt}}.

Both parts are non-negative, so ``H`` stays non-negative. Unlike the baseline
KL update, this gradient-split rule is not proven to decrease ``J`` at every
step; descent is checked empirically (see ``tests/test_modified.py``).

References
----------
T. Virtanen, "Monaural sound source separation by nonnegative matrix
factorization with temporal continuity and sparseness criteria," IEEE
Transactions on Audio, Speech, and Language Processing, 15(3), 1066-1074, 2007.
"""

import numpy as np

from src.baseline import kl_divergence


def _check_mu(mu):
    """Raise ``ValueError`` unless ``mu`` is a finite, non-negative number."""
    if not np.isfinite(mu) or mu < 0:
        raise ValueError("mu must be finite and non-negative.")


def neighbour_sums(H):
    """Sum and count the temporal neighbours of every activation entry.

    Parameters
    ----------
    H : array_like, shape (K, T)
        Activations.

    Returns
    -------
    dict
        ``"N"`` : ndarray, shape (K, T) -- ``H[:, t-1] + H[:, t+1]``, using only
        neighbours that exist (no padding beyond the clip).
        ``"c"`` : ndarray, shape (T,) -- number of neighbours of each frame:
        2 inside, 1 at the ends, 0 if ``T == 1``.
    """
    H = np.asarray(H, dtype=float)
    T = H.shape[1]

    N = np.zeros_like(H)
    N[:, 1:] += H[:, :-1]
    N[:, :-1] += H[:, 1:]

    c = np.full(T, 2.0)
    if T >= 1:
        c[0] -= 1.0
        c[-1] -= 1.0
    c = np.maximum(c, 0.0)

    return {
        "N": N,
        "c": c
    }


def smoothness_penalty(H, mu):
    """Temporal smoothness penalty ``(mu / 2) * sum_k sum_t (H_kt - H_k,t-1)^2``.

    Parameters
    ----------
    H : array_like, shape (K, T)
        Activations.
    mu : float
        Penalty weight (must be finite and >= 0).

    Returns
    -------
    float
        The penalty. Zero for ``mu == 0``, for ``T <= 1`` and for activations
        that are constant in time.
    """
    _check_mu(mu)
    H = np.asarray(H, dtype=float)
    diffs = np.diff(H, axis=1)
    return float(0.5 * mu * np.sum(diffs ** 2))


def smooth_objective(V, W, H, mu, eps=1e-10):
    """Modified objective ``J = D(V || WH) + smoothness_penalty(H, mu)``.

    Parameters
    ----------
    V : array_like, shape (F, T)
        Non-negative target matrix.
    W : array_like, shape (F, K)
        Templates.
    H : array_like, shape (K, T)
        Activations.
    mu : float
        Penalty weight (must be finite and >= 0).
    eps : float, optional
        Floor on the reconstruction inside the KL term.

    Returns
    -------
    dict
        ``"total"`` : float -- ``J``.
        ``"kl"`` : float -- the KL term ``D(V || WH)``.
        ``"penalty"`` : float -- the smoothness term.
    """
    W = np.asarray(W, dtype=float)
    H = np.asarray(H, dtype=float)
    kl = kl_divergence(V, W @ H, eps)
    pen = smoothness_penalty(H, mu)
    return {
        "total": kl + pen,
        "kl": kl,
        "penalty": pen
    }


def update_activations_smooth(V, W, H, mu, eps=1e-10):
    """Apply one smoothness-penalized multiplicative update to H.

    .. math::

        H \\leftarrow H \\odot
            \\frac{W^T (V / WH) + \\mu N}{W^T \\mathbf{1} + \\mu c \\odot H}

    See the module docstring for ``N`` and ``c``. With ``mu == 0`` this is
    exactly :func:`src.baseline.update_activations`.

    Parameters
    ----------
    V : array_like, shape (F, T)
        Non-negative target matrix.
    W : array_like, shape (F, K)
        Current templates (held fixed).
    H : array_like, shape (K, T)
        Current activations.
    mu : float
        Penalty weight (must be finite and >= 0).
    eps : float, optional
        Floor for the reconstruction and denominator to avoid division by zero.

    Returns
    -------
    ndarray, shape (K, T)
        Updated activations. The input ``H`` is not modified.
    """
    _check_mu(mu)
    V = np.asarray(V)
    W = np.asarray(W)
    H = np.asarray(H)

    Q = V / np.maximum(W @ H, eps)
    num = W.T @ Q
    den = W.sum(axis=0)[:, None]

    if mu > 0:
        nb = neighbour_sums(H)
        num = num + mu * nb["N"]
        den = den + mu * nb["c"][None, :] * H

    H_new = H * (num / np.maximum(den, eps))

    return H_new


def _init_activations(V, W, seed):
    """Initial H, identical to :func:`src.baseline.infer_activations`.

    ``H0 ~ U(0.5, 1.5)`` from ``default_rng(seed)``, scaled so that
    ``mean(W @ H0) == mean(V)``. Baseline and modified inference therefore start
    from the same positive values for the same seed.
    """
    rng = np.random.default_rng(seed)
    T = V.shape[1]
    K = W.shape[1]
    H0 = rng.uniform(0.5, 1.5, size=(K, T))
    return H0 * np.mean(V) / np.mean(W @ H0)


def infer_activations_smooth(V, W, mu, n_iter=200, seed=0, eps=1e-10, track_loss=True):
    """Estimate activations H for V with fixed W under the smoothness penalty.

    Same initialization as :func:`src.baseline.infer_activations`, then
    ``n_iter`` applications of :func:`update_activations_smooth`. With
    ``mu == 0`` the returned ``H`` equals the baseline's.

    Parameters
    ----------
    V : ndarray, shape (F, T)
        Non-negative, finite matrix to decompose.
    W : ndarray, shape (F, K)
        Fixed templates, e.g. the aligned dictionary from
        :func:`src.protocol.learn_dictionary`.
    mu : float
        Penalty weight (must be finite and >= 0).
    n_iter : int, optional
        Number of update iterations (must be >= 0).
    seed : int, optional
        Seed for the random initialization of ``H``.
    eps : float, optional
        Numerical floor used in the updates and the loss.
    track_loss : bool, optional
        If True, record ``J``, the KL term and the penalty after
        initialization and after every iteration.

    Returns
    -------
    dict
        ``"H"`` : ndarray, shape (K, T) -- inferred activations.
        ``"loss"`` : ndarray -- ``J`` per iteration, shape (n_iter + 1,), or an
        empty array if not ``track_loss``.
        ``"kl"`` : ndarray -- KL term, same shape as ``"loss"``.
        ``"penalty"`` : ndarray -- smoothness term, same shape as ``"loss"``.
        ``"mu"`` : float -- the penalty weight used.

    Raises
    ------
    ValueError
        If ``W`` and ``V`` have a different number of rows, ``n_iter < 0``,
        ``mu`` is negative or not finite, or ``V`` has negative or non-finite
        entries.
    """
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    _check_mu(mu)
    if W.shape[0] != V.shape[0]:
        raise ValueError("W.shape[0] != V.shape[0]")
    if n_iter < 0:
        raise ValueError("n_iter must be >= 0.")
    if np.any(V < 0) or not np.all(np.isfinite(V)):
        raise ValueError("V must have all non-negative and finite values.")

    H = _init_activations(V, W, seed)

    size = n_iter + 1 if track_loss else 0
    loss, kl, pen = np.zeros(size), np.zeros(size), np.zeros(size)

    def record(i):
        obj = smooth_objective(V, W, H, mu, eps)
        loss[i], kl[i], pen[i] = obj["total"], obj["kl"], obj["penalty"]

    if track_loss:
        record(0)

    for i in range(n_iter):
        H = update_activations_smooth(V, W, H, mu, eps)
        if track_loss:
            record(i + 1)

    return {
        "H": H,
        "loss": loss,
        "kl": kl,
        "penalty": pen,
        "mu": float(mu)
    }
