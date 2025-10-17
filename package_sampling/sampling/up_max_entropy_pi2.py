"""
Joint inclusion probabilities (π₂) for UP-Maximum-Entropy sampling.

This is a faithful, *practical* port of R’s `UPmaxentropypi2()` from the
*sampling* package (v 2.10).

Key points
----------
* For **n ≥ 2** it follows the same block–logic as the R routine:

      – extract live units (0 < πᵢ < 1);
      – compute an adjusted π̃ via the iterative MaxEnt routine;
      – transform to weights w = π̃ / (1 − π̃);
      – call `upme_pik2_from_pik_w()` for the live block
        (implemented here, vectorised).

* Deterministic units (πᵢ≈1) get whole rows/columns = π.

* For the **n = 1** pathological case the π₂ matrix is simply diagonal.

* The function never guesses: shapes and diagonals are guaranteed
  correct and all off-diagonal cells respect the upper bound
  πᵢⱼ ≤ min(πᵢ, πⱼ).
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from package_sampling.utils import as_int

from .upme_pik2_from_pik_w import upme_pik2_from_pik_w
from .upme_pik_tilde_from_pik import upme_pik_tilde_from_pik


def up_max_entropy_pi2(
    pik: Union[List[float], NDArray[np.floating]],
    eps: float = 1e-6,
) -> NDArray[np.floating]:
    if not isinstance(pik, np.ndarray):
        pik = np.asarray(pik, dtype=float)

    n = as_int(round(pik.sum()))
    N = pik.size
    M = np.zeros((N, N))

    if n >= 2:
        live = (pik > eps) & (pik < 1 - eps)
        pik_live = pik[live]

        pik_tilde = upme_pik_tilde_from_pik(pik_live, eps=eps)
        w = pik_tilde / (1.0 - pik_tilde)
        M_live = upme_pik2_from_pik_w(pik_live, w)
        idx = np.flatnonzero(live)
        M[np.ix_(idx, idx)] = M_live

        if np.any(pik >= 1 - eps):
            M[:, pik >= 1 - eps] = pik
            M[pik >= 1 - eps, :] = pik[None, :]

    elif n == 1:
        np.fill_diagonal(M, pik)

    lim = np.minimum.outer(pik, pik)
    M = np.minimum(M, lim)
    np.fill_diagonal(M, pik)

    return M
