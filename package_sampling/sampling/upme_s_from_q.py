from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def upme_s_from_q(
    q: NDArray[np.floating],
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int8]:
    """
    Draw a 0/1 sample vector from *q* (R `UPMEsfromq`).

    Parameters
    ----------
    q : ndarray, shape (N, n)
        Conditional probability matrix.
    rng : numpy.random.Generator or None
        Random source; defaults to ``np.random.default_rng()``.

    Returns
    -------
    s : ndarray[int8], shape (N,)
        Selected units (1) vs non-selected (0).
    """
    if not isinstance(q, np.ndarray) or q.ndim != 2:
        raise ValueError("`q` must be a 2-D NumPy array.")
    N, n = q.shape
    s = np.zeros(N, dtype=np.int8)
    if n == 0:
        return s

    rng = rng or np.random.default_rng()
    remaining = n
    for k in range(N):
        if remaining and rng.random() < q[k, remaining - 1]:
            s[k] = 1
            remaining -= 1
            if remaining == 0:
                break

    return s
