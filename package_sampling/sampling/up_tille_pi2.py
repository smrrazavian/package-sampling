"""
Joint inclusion probabilities for Tillé sampling (π₂ matrix).

Computes an N*N matrix `pi2` where:
  * diagonal = first-order π
  * off-diagonal (i<j) = π_ij
For large N this is O(N²) memory; use sparsity tricks if needed.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from package_sampling.utils import as_int, inclusion_probabilities


def up_tille_pi2(
    pik: Union[List[float], NDArray[np.floating]],
    eps: float = 1e-6,
) -> NDArray[np.floating]:
    if not isinstance(pik, np.ndarray):
        pik = np.asarray(pik, dtype=float)

    live = (pik > eps) & (pik < 1 - eps)
    pik_live = pik[live]
    N = pik_live.size
    if N == 0:
        raise ValueError("All inclusion probs are outside ]eps,1-eps[ .")

    n = as_int(pik_live.sum())
    if n == N:  # full census
        return np.ones((pik.size, pik.size))

    UN = np.ones(N)
    b_prev = np.ones(N)
    P = np.ones((N, N))

    for k in range(N - n):
        a = inclusion_probabilities(pik_live, N - k - 1)
        v = 1.0 - a / b_prev
        b_prev = a
        d = np.outer(v, UN)
        P *= 1.0 - d - d.T

    np.fill_diagonal(P, pik_live)

    big = np.zeros((pik.size, pik.size))
    idx = np.flatnonzero(live)
    big[np.ix_(idx, idx)] = P
    return big
