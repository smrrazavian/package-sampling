"""
Tillé unequal-probability sampling without replacement (fixed size).

Reference:
  Deville & Tillé (1998) “Unequal probability sampling without replacement
  through a splitting method”, *Biometrika* 85 : 89-101.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Union

from package_sampling.utils import as_int, inclusion_probabilities


def up_tille(
    pik: Union[List[float], NDArray[np.floating]],
    eps: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int8]:
    if not isinstance(pik, np.ndarray):
        pik = np.asarray(pik, dtype=float)

    if np.any(np.isnan(pik)):
        raise ValueError("`pik` contains NaN values.")

    live = (pik > eps) & (pik < 1 - eps)
    pik_live = pik[live]
    N = pik_live.size
    if N == 0:
        raise ValueError("All inclusion probs are outside ]eps,1-eps[ .")

    n = as_int(pik_live.sum())
    if n == 0:
        out = np.where(pik >= 1 - eps, 1, 0).astype(np.int8)
        return out

    if n == N:
        out = np.zeros_like(pik, dtype=np.int8)
        out[pik >= 1 - eps] = 1
        out[live] = 1
        return out

    rng = rng or np.random.default_rng()
    sb = np.ones(N, dtype=np.int8)
    prev_b = np.ones(N, dtype=float)

    for k in range(N - n):
        a = inclusion_probabilities(pik_live, N - k - 1)
        v = 1.0 - a / prev_b
        prev_b = a

        probs = v * sb
        total = probs.sum()
        if total <= 0:
            idx = rng.choice(np.flatnonzero(sb))
        else:
            idx = np.searchsorted(np.cumsum(probs) / total, rng.random())
        sb[idx] = 0

    out = np.zeros_like(pik, dtype=np.int8)
    out[pik >= 1 - eps] = 1
    out[live] = sb
    return out
