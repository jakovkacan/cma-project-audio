from time import perf_counter

import numpy as np


def time_call(fn, *args, warmup=1, repeats=5, **kwargs):
    """Time ``fn(*args, **kwargs)`` with ``time.perf_counter``.

    ``fn`` is called ``warmup`` times without recording, then ``repeats`` times
    with timing. Returns a dict with ``"median_s"``, ``"min_s"``, ``"all_s"``
    (list of per-call wall times in seconds) and ``"result"`` (return value of
    the last call).
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    for _ in range(warmup):
        fn(*args, **kwargs)

    times = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = fn(*args, **kwargs)
        times.append(perf_counter() - start)

    return {
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
        "all_s": times,
        "result": result,
    }
