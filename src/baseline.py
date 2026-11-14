"""Baseline non-negative matrix factorization (NMF) with KL divergence.

Implements the multiplicative update rules of Lee & Seung (2001) for the
generalized Kullback-Leibler divergence. A non-negative matrix ``V`` of shape
``(F, T)`` (e.g. a magnitude spectrogram with ``F`` frequency bins and ``T``
frames) is approximated as ``V ≈ W @ H``, where ``W`` of shape ``(F, K)`` holds
the templates and ``H`` of shape ``(K, T)`` holds the activations.
"""

import numpy as np
from scipy.special import xlogy


def kl_divergence(V, Z, eps=1e-10):
    """Compute the generalized Kullback-Leibler divergence D(V || Z).

    .. math:: D(V \\| Z) = \\sum_{f,t} V \\log(V / Z) - V + Z

    Terms where ``V == 0`` contribute ``Z`` only (``0 * log 0`` is taken as 0).

    Parameters
    ----------
    V : array_like
        Non-negative target matrix.
    Z : array_like
        Approximation of ``V``, same shape as ``V``.
    eps : float, optional
        Floor applied to ``Z`` to avoid division by zero and ``log(0)``.

    Returns
    -------
    float
        The divergence, summed over all entries.

    Raises
    ------
    ValueError
        If ``V`` and ``Z`` differ in shape, or ``V`` has negative entries.
    """
    V = np.asarray(V, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if V.shape != Z.shape:
        raise ValueError("V and Z shape mismatch.")
    if np.any(V < 0):
        raise ValueError("All entries of V must be non-negative.")

    Z = np.maximum(Z, eps)

    D = np.sum(xlogy(V, V / Z) - V + Z)

    return float(D)


def init_factors(V, n_components, seed):
    """Randomly initialize the NMF factors W and H.

    Entries are drawn uniformly from ``[0.5, 1.5)`` and then both factors are
    scaled by the same factor ``alpha`` so that ``mean(W @ H) == mean(V)``.

    Parameters
    ----------
    V : array_like, shape (F, T)
        Non-negative matrix to be factorized; used for its shape and scale.
    n_components : int
        Number of components ``K``.
    seed : int
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    dict
        ``"W"`` : ndarray, shape (F, K) -- initial templates.
        ``"H"`` : ndarray, shape (K, T) -- initial activations.
    """
    rng = np.random.default_rng(seed)
    V = np.asarray(V)
    (F, T) = V.shape
    K = n_components

    W0 = rng.uniform(0.5, 1.5, size=(F, K))
    H0 = rng.uniform(0.5, 1.5, size=(K, T))

    alpha = np.sqrt(np.mean(V) / np.mean(W0 @ H0))
    W = alpha * W0
    H = alpha * H0

    return {
        "W": W,
        "H": H
    }


def update_activations(V, W, H, eps=1e-10):
    """Apply one KL multiplicative update to the activations H.

    .. math:: H \\leftarrow H \\odot \\frac{W^T (V / WH)}{W^T \\mathbf{1}}

    Parameters
    ----------
    V : array_like, shape (F, T)
        Non-negative target matrix.
    W : array_like, shape (F, K)
        Current templates (held fixed).
    H : array_like, shape (K, T)
        Current activations.
    eps : float, optional
        Floor for the reconstruction and denominator to avoid division by zero.

    Returns
    -------
    ndarray, shape (K, T)
        Updated activations. The input ``H`` is not modified.
    """
    V = np.asarray(V)
    W = np.asarray(W)
    H = np.asarray(H)

    Q = V / (np.maximum((W @ H), eps))
    den = np.maximum(W.sum(axis=0)[:, None], eps)
    H_new = H * ((W.T @ Q) / den)

    return H_new


def update_templates(V, W, H, eps=1e-10):
    """Apply one KL multiplicative update to the templates W.

    .. math:: W \\leftarrow W \\odot \\frac{(V / WH) H^T}{\\mathbf{1} H^T}

    Parameters
    ----------
    V : array_like, shape (F, T)
        Non-negative target matrix.
    W : array_like, shape (F, K)
        Current templates.
    H : array_like, shape (K, T)
        Current activations (held fixed).
    eps : float, optional
        Floor for the reconstruction and denominator to avoid division by zero.

    Returns
    -------
    ndarray, shape (F, K)
        Updated templates. The input ``W`` is not modified.
    """
    V = np.asarray(V)
    W = np.asarray(W)
    H = np.asarray(H)

    Q = V / (np.maximum((W @ H), eps))
    den = np.maximum(H.sum(axis=1)[None, :], eps)
    W_new = W * ((Q @ H.T) / den)

    return W_new


def normalize_templates(W, H, eps=1e-10):
    """Rescale so each template (column of W) sums to 1.

    Each column of ``W`` is divided by its sum and the matching row of ``H`` is
    multiplied by the same amount, so the product ``W @ H`` is unchanged. This
    removes the scale ambiguity between the two factors.

    Parameters
    ----------
    W : array_like, shape (F, K)
        Templates.
    H : array_like, shape (K, T)
        Activations.
    eps : float, optional
        Floor on the column sums to avoid division by zero.

    Returns
    -------
    dict
        ``"W"`` : ndarray, shape (F, K) -- column-normalized templates.
        ``"H"`` : ndarray, shape (K, T) -- correspondingly rescaled activations.
    """
    W = np.asarray(W)
    H = np.asarray(H)

    s = np.maximum(W.sum(axis=0), eps)

    W_new = W / s[None, :]
    H_new = H * s[:, None]

    return {
        "W": W_new,
        "H": H_new
    }


def fit_nmf(V, n_components, n_iter=200, seed=0, eps=1e-10, track_loss=True):
    """Factorize V ≈ W @ H with KL-divergence NMF.

    Each iteration updates ``H``, then ``W``, then normalizes the columns of
    ``W`` to sum to 1 (see :func:`normalize_templates`).

    Parameters
    ----------
    V : ndarray, shape (F, T)
        Non-negative, finite matrix to factorize.
    n_components : int
        Number of components ``K`` (must be >= 1).
    n_iter : int, optional
        Number of update iterations (must be >= 0).
    seed : int, optional
        Seed for the random initialization.
    eps : float, optional
        Numerical floor used in the updates and the loss.
    track_loss : bool, optional
        If True, compute the KL divergence after initialization and after
        every iteration.

    Returns
    -------
    dict
        ``"W"`` : ndarray, shape (F, K) -- learned templates (columns sum to 1).
        ``"H"`` : ndarray, shape (K, T) -- learned activations.
        ``"loss"`` : ndarray -- shape (n_iter + 1,) with ``loss[0]`` the initial
        loss if ``track_loss``, otherwise an empty array.
        ``"seed"`` : int -- seed used.
        ``"n_iter"`` : int -- number of iterations run.

    Raises
    ------
    ValueError
        If ``n_components < 1``, ``n_iter < 0``, or ``V`` contains negative or
        non-finite values.
    """
    if n_components < 1 or n_iter < 0:
        raise ValueError("n_components < 1 or n_iter < 0.")
    if np.any(V < 0) or not np.all(np.isfinite(V)):
        raise ValueError("V must have all non-negative and finite values.")

    factors = init_factors(V, n_components, seed)
    W = factors["W"]
    H = factors["H"]
    Z = W @ H

    loss = np.zeros(n_iter + 1) if track_loss else np.array([])
    if track_loss: loss[0] = kl_divergence(V, Z, eps)

    for i in range(n_iter):
        H = update_activations(V, W, H, eps)
        W = update_templates(V, W, H, eps)

        norm = normalize_templates(W, H, eps)
        H = norm["H"]
        W = norm["W"]

        if not track_loss:
            continue

        Z = W @ H
        loss[i + 1] = kl_divergence(V, Z, eps)

    return {
        "W": W,  # (F, K)
        "H": H,  # (K, T)
        "loss": loss,  # n_iter + 1 float array or []
        "seed": int(seed),
        "n_iter": int(n_iter)
    }


def fit_nmf_restarts(V, n_components, seeds, n_iter=300, eps=1e-10):
    """Run :func:`fit_nmf` from several seeds and keep the best run.

    The best run is the one with the lowest final KL divergence. Ties keep the
    earliest seed.

    Parameters
    ----------
    V : ndarray, shape (F, T)
        Non-negative, finite matrix to factorize.
    n_components : int
        Number of components ``K``.
    seeds : array_like of int
        Seeds to try; must be non-empty.
    n_iter : int, optional
        Number of iterations per run.
    eps : float, optional
        Numerical floor passed to :func:`fit_nmf`.

    Returns
    -------
    dict
        ``"W"``, ``"H"``, ``"loss"`` : results of the best run (see
        :func:`fit_nmf`).
        ``"best_seed"`` : int -- seed of the best run.
        ``"seed"`` : int -- same as ``"best_seed"``, for compatibility with
        :func:`fit_nmf` output.
        ``"final_losses"`` : ndarray, shape (len(seeds),) -- final loss of each
        run, in the order of ``seeds``.
        ``"n_iter"`` : int -- iterations per run.
    """
    seeds = np.asarray(seeds)
    if seeds.size == 0:
        raise ValueError("seeds must not be empty.")
    losses = np.zeros(seeds.size)

    best_loss = np.inf
    best_run = {}
    best_seed = 0

    for i in range(seeds.size):
        run = fit_nmf(V, n_components, n_iter, seeds[i], eps)
        loss = run["loss"]
        losses[i] = loss[-1]

        if loss[-1] < best_loss:
            best_run = run
            best_seed = seeds[i]
            best_loss = loss[-1]

    return {
        "W": best_run["W"],
        "H": best_run["H"],
        "loss": best_run["loss"],
        "best_seed": int(best_seed),
        "seed": int(best_seed),
        "final_losses": losses,
        "n_iter": int(n_iter)
    }


def infer_activations(V, W, n_iter=200, seed=0, eps=1e-10, track_loss=True):
    """Estimate activations H for new data V with fixed templates W.

    ``H`` is initialized uniformly in ``[0.5, 1.5)``, scaled so that
    ``mean(W @ H) == mean(V)``, then refined with KL multiplicative updates
    while ``W`` stays fixed.

    Parameters
    ----------
    V : ndarray, shape (F, T)
        Non-negative matrix to decompose.
    W : ndarray, shape (F, K)
        Fixed templates, e.g. from :func:`fit_nmf`.
    n_iter : int, optional
        Number of update iterations.
    seed : int, optional
        Seed for the random initialization of ``H``.
    eps : float, optional
        Numerical floor used in the updates and the loss.
    track_loss : bool, optional
        If True, compute the KL divergence after initialization and after
        every iteration.

    Returns
    -------
    dict
        ``"H"`` : ndarray, shape (K, T) -- inferred activations.
        ``"loss"`` : ndarray -- shape (n_iter + 1,) if ``track_loss``,
        otherwise an empty array.

    Raises
    ------
    ValueError
        If ``W`` and ``V`` have a different number of rows.
    """
    if W.shape[0] != V.shape[0]:
        raise ValueError("W.shape[0] != V.shape[0]")

    rng = np.random.default_rng(seed)
    T = V.shape[1]
    K = W.shape[1]
    H0 = rng.uniform(0.5, 1.5, size=(K, T))
    H0 = H0 * np.mean(V) / np.mean(W @ H0)
    Z = W @ H0

    loss = np.zeros(n_iter + 1) if track_loss else np.array([])
    if track_loss: loss[0] = kl_divergence(V, Z, eps)

    for i in range(n_iter):
        H0 = update_activations(V, W, H0, eps=1e-10)

        if not track_loss:
            continue

        Z = W @ H0
        loss[i + 1] = kl_divergence(V, Z, eps)

    return {
        "H": H0,
        "loss": loss
    }
