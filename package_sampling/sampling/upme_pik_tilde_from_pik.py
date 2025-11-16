"""
Iterative “tilde” adjustment of inclusion probabilities for the
Maximum-Entropy design (R `UPMEpiktildefrompik`).

Given a target vector ``pik`` the algorithm finds π̃ such that

    * Σ π̃ = Σ pik   (fixed sample size n),
    * π̃ stays in (0,1),
    * π̃ is consistent with the Max-Entropy sequential q-matrix.

The fixed-point iteration stops when
‖π̃(k+1) – π̃(k)‖₁ ≤ *eps*.
"""

from __future__ import annotations

from typing import Any, List, Union, cast

import numpy as np
from numpy.typing import NDArray

from package_sampling.utils import as_int

from .upme_pik_from_q import upme_pik_from_q
from .upme_q_from_w import upme_q_from_w


def upme_pik_tilde_from_pik(
    pik: Union[List[float], NDArray[np.floating]],
    eps: float = 1e-6,
    max_iter: int = 1_000,
) -> NDArray[np.floating]:
    """
    Parameters
    ----------
    pik : 1-D array-like
        Target first-order inclusion probabilities (0 ≤ πᵢ ≤ 1).
    eps : float, default 1e-6
        Convergence tolerance on the L¹-difference between consecutive π̃.
    max_iter : int, default 1000
        Hard cap on iterations (avoids infinite loops if *eps* too tight).

    Returns
    -------
    pik_tilde : ndarray[float]
        Adjusted inclusion probabilities π̃.

    Raises
    ------
    RuntimeError
        If the algorithm fails to converge within `max_iter`.
    """
    # ---------- validation -------------------------------------------------
    pik = cast(NDArray[np.floating[Any]], np.asarray(pik, dtype=float))
    if pik.ndim != 1:
        raise ValueError("`pik` must be 1-D.")
    if pik.size == 0 or np.all(pik == 0):
        return np.zeros_like(pik)
    if np.any(np.isnan(pik)):
        raise ValueError("`pik` must not contain NaN.")

    pik = np.clip(pik, 0.0, 1.0)
    n = as_int(pik.sum())
    if n == 0:
        return np.zeros_like(pik)
    if n == pik.size:
        return np.ones_like(pik)

    # ---------- main fixed-point loop -------------------------------------
    pik_t = pik.copy()
    for _ in range(max_iter):
        pik_t_safe = np.clip(pik_t, eps, 1.0 - eps)
        w = pik_t_safe / (1.0 - pik_t_safe)

        q = upme_q_from_w(w, n)
        pik_from_q = upme_pik_from_q(q)

        pik_next = np.clip(pik_t + pik - pik_from_q, 0.0, 1.0)

        change = np.abs(pik_next - pik_t).sum()
        if change <= eps:
            return pik_next

        pik_t = pik_next

    raise RuntimeError(
        "upme_pik_tilde_from_pik: did not converge "
        f"after {max_iter} iterations (last Δ={change:.2g})."
    )
