"""
Joint inclusion probabilities for UP Systematic sampling (π₂ matrix).

One-for-one port of R's `UPsystematicpi2` (package *sampling* 2.10).
Complexity: O(N²) memory, but only a few dense matrix ops.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Union


def up_systematic_pi2(
    pik: Union[List[float], NDArray[np.floating]],
) -> NDArray[np.floating]:
    if not isinstance(pik, np.ndarray):
        pik = np.asarray(pik, dtype=float)

    # deterministic 0/1 units are treated later
    pik_live = pik[(pik > 0) & (pik < 1)]
    N = pik_live.size
    if N == 0:
        return np.outer(pik, pik)

    # ------- build centroids & segment lengths -------------------------
    Vk = np.cumsum(pik_live)
    Vk1 = Vk % 1
    if Vk1[-1] != 0:
        Vk1[-1] = 0.0
    r = np.sort(Vk1)
    r = np.concatenate((r, [1.0]))
    cent = 0.5 * (r[:-1] + r[1:])
    p = np.diff(r)

    # ------- incidence matrix M ---------------------------------------
    # A_{ij} = ((0, cumsum(pik_live)) - cent_j) mod 1
    csum = np.concatenate(([0.0], Vk))
    A = (csum[:, None] - cent) % 1.0
    M = (A[:-1, :] > A[1:, :]).astype(int)

    pi21 = M @ np.diag(p) @ M.T

    # ------- embed back into full N_pop × N_pop matrix -----------------
    pi2 = np.outer(pik, pik)
    live_idx = np.where((pik > 0) & (pik < 1))[0]
    for i, ii in enumerate(live_idx):
        pi2[ii, live_idx] = pi21[i]
        pi2[live_idx, ii] = pi21[:, i]
    return pi2
