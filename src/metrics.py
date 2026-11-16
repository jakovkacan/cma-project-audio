import numpy as np
from src.baseline import kl_divergence


def frame_f1(pred, true):
    pred = np.asarray(pred)
    true = np.asarray(true)

    if pred.shape != true.shape:
        raise ValueError("pred and true must be of same shape.")

    tp = np.count_nonzero(true & pred)  # True - True
    fp = np.count_nonzero(~true & pred)  # False - True
    fn = np.count_nonzero(true & ~pred)  # True - False

    den = (2 * tp + fp + fn)
    f1 = (2 * tp) / den if den != 0 else 0

    return {
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn)
    }


def kl_per_entry(V, W, H, eps=1e-10):
    V = np.asarray(V)
    W = np.asarray(W)
    H = np.asarray(H)
    Z = W @ H

    D = kl_divergence(V, Z, eps)

    return float(D / V.size)


def onset_lags(pred, true, max_lag=8):
    pred = np.asarray(pred).flatten()
    true = np.asarray(true).flatten()

    if pred.shape != true.shape:
        raise ValueError("pred and true must be of same shape.")

    n = len(true)
    true_onset_indices = []
    pred_onset_indices = []
    l = []
    missed = 0

    if true[0]:
        true_onset_indices.append(0)

    for i in range(1, n):
        if true[i] and not true[i - 1]:
            true_onset_indices.append(i)

    for i in range(len(true_onset_indices)):
        for j in range(max(0, true_onset_indices[i] - max_lag), min(n, true_onset_indices[i] + max_lag)):
            if pred[j]:
                pred_onset_indices.append(j)
                l.append(j - true_onset_indices[i])
                break

        missed += 1

    return {
        "lags": np.array(l, dtype=int),
        "missed": missed,
        "n_onsets": len(true_onset_indices)
    }
