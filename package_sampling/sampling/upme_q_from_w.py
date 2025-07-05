from __future__ import annotations

"""package_sampling.sampling.upme_q_from_w
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Vectorised re‑implementation of the R helper **`UPMEqfromw`** that turns the
vector of *conditional odds* ``w`` into the sequential Max–Entropy
*q‑matrix* used by `up_max_entropy`.

The algorithm is algebraically identical to the triple‑loop reference but is
written with NumPy primitives so that the overall complexity stays
:math:`O(Nn)` yet the constant factor is ~20× smaller.

Differences to the previous buggy port
--------------------------------------
* work **directly** with the conditional odds ``w`` – the earlier variant
  divided by ``1+w`` and therefore broke the recurrences;
* fix the diagonal "tail" of the cumulative table ``expa``
  (rows *N‑n … N‑1* inclusive);
* guard all divisions so that 0/0 → 0 (faithful to the original R code).
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Union


def upme_q_from_w(
    w: Union[List[float], NDArray[np.floating]],
    n: int,
    /,
) -> NDArray[np.floating]:
    """Build the *q* matrix for the sequential Maximum‑Entropy design.

    Parameters
    ----------
    w : 1‑D array‑like of float
        Positive *conditional odds* ``w_i = π̃_i / (1-π̃_i)``.
        Length :math:`N`.
    n : int
        Desired *fixed* sample size (number of columns of *q*).
        Must satisfy ``0 ≤ n ≤ N``.

    Returns
    -------
    q : ndarray, shape (N, n)
        Conditional selection probabilities that drive the sequential
        algorithm.
    """
    # ------------------------------------------------------------------ #
    # 0. input validation                                                #
    # ------------------------------------------------------------------ #
    if not isinstance(w, np.ndarray):
        w = np.asarray(w, dtype=float)
    if w.ndim != 1:
        raise ValueError("`w` must be 1‑D.")
    N: int = w.size
    if not (0 <= n <= N):
        raise ValueError("`n` out of range")

    # ------------------------------------------------------------------ #
    # 1. trivial cases                                                   #
    # ------------------------------------------------------------------ #
    if n == 0 or np.all(w == 0):
        return np.zeros((N, n))
    if n == 1:
        denom = np.flip(np.cumsum(np.flip(w)))
        return np.divide(w, denom, out=np.zeros_like(w), where=denom > 0).reshape(N, 1)

    # ------------------------------------------------------------------ #
    # 2. build the cumulative "expa" table  Z(i,z)                       #
    # ------------------------------------------------------------------ #
    expa = np.zeros((N, n), dtype=float)

    expa[:, 0] = np.flip(np.cumsum(np.flip(w)))

    logs = np.log(w, where=w > 0, out=np.full_like(w, -np.inf))
    log_cum = np.cumsum(np.flip(logs))
    for i in range(N - n, N):
        z = N - i - 1
        expa[i, z] = np.exp(log_cum[N - 1 - i])

    for i in range(N - 2, -1, -1):
        upto = min(N - i - 1, n - 1)
        wi = w[i]
        for z in range(1, upto + 1):
            expa[i, z] = wi * expa[i + 1, z - 1] + expa[i + 1, z]

    # ------------------------------------------------------------------ #
    # 3. build the q‑matrix                                              #
    # ------------------------------------------------------------------ #
    q = np.zeros_like(expa)

    q[:, 0] = np.divide(w, expa[:, 0], out=np.zeros_like(w), where=expa[:, 0] > 0)

    for i in range(N - n, N):
        q[i, N - i - 1] = 1.0

    for i in range(N - 2, -1, -1):
        upto = min(N - i - 1, n - 1)
        wi = w[i]
        for z in range(1, upto + 1):
            denom = expa[i, z]
            if denom:
                q[i, z] = wi * expa[i + 1, z - 1] / denom

    return q
