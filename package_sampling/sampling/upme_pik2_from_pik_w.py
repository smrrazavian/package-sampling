from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from package_sampling.utils import as_int


def upme_pik2_from_pik_w(pik: NDArray, w: NDArray, eps: float = 1e-12) -> NDArray:
    """
    UPME joint-inclusion block  π₂  ⇐  (π , w)

    Vectorised 1-for-1 translation of R’s  `UPMEpik2frompikw()`.

    Parameters
    ----------
    pik : 1-D ndarray of live first-order π  (0 < π < 1)
    w   : 1-D ndarray with  w = π / (1 − π)  same length as `pik`

    Returns
    -------
    square ndarray (N×N) containing the joint probabilities for the live units.
    Diagonal equals π.
    """
    pik = np.asarray(pik, dtype=float)
    w = np.asarray(w, dtype=float)
    if pik.shape != w.shape or pik.ndim != 1:
        raise ValueError("`pik` and `w` must be 1-D arrays of the same length")

    N = pik.size
    n = as_int(round(pik.sum()))
    M = np.empty((N, N), dtype=float)

    # --- first pass: formula wherever π_k ≠ π_l ---------------------------
    diff = pik[:, None] != pik[None, :]
    denom = w[None, :] - w[:, None]

    with np.errstate(divide="ignore", invalid="ignore"):
        M[:] = (pik[:, None] * w[None, :] - pik[None, :] * w[:, None]) / denom
    M[~diff] = np.nan
    np.fill_diagonal(M, pik)

    for k in range(N):
        undef = np.isnan(M[k])
        comp = int(undef.sum())
        if comp == 0:
            continue
        tt = np.nansum(M[k])
        cc = (n * pik[k] - tt) / comp
        M[k, undef] = cc

    lim = np.minimum.outer(pik, pik) + eps
    M = np.clip(M, 0.0, lim)
    np.fill_diagonal(M, pik)
    return M
