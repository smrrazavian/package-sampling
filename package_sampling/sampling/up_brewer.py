"""
Brewer unequal-probability sampling without replacement (fixed size).

Reference: Brewer & Hanif (1983), “Sampling with unequal probabilities”.
Exact port of R’s `UPbrewer` from package *sampling* 2.10,
but vectorised and exposing an `rng` parameter.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from package_sampling.utils import as_int


def up_brewer(
    pik: Union[List[float], NDArray[np.floating]],
    eps: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int8]:
    """
    Brewer unequal-probability sampling (fixed-size, without replacement).

    Implements the Brewer–Hanif algorithm (Brewer & Hanif 1983, §8.5) as in
    *sampling* 2.10’s **UPbrewer**.  All “live” units
    (``eps < πᵢ < 1−eps``) are treated in a sequential ‘take-one’ loop; units
    with π≈0 or π≈1 are handled deterministically.

    Parameters
    ----------
    pik : 1-D array-like of float
        First-order inclusion probabilities (0 ≤ πᵢ ≤ 1).
    eps : float, default ``1e-6``
        Threshold separating deterministic and live units:
        * πᵢ ≤ *eps* → never selected,
        * πᵢ ≥ 1−*eps* → always selected.
    rng : numpy.random.Generator or None, default *None*
        Random-number generator.  Falls back to ``np.random.default_rng()`` if
        *None*.

    Returns
    -------
    out : ndarray of int8
        0/1 mask (length ``len(pik)``) indicating selected units.
        The number of ones equals ``round(sum(pik_live))``.

    Raises
    ------
    ValueError
        If *pik* contains NaNs, or if every πᵢ lies outside
        ``(eps, 1−eps)``.

    Notes
    -----
    **Algorithm.**

    Let *n* = ``round(sum(pik_live))``.  At step *i* (1 ≤ *i* ≤ *n*) with
    current selection vector ``sb``:

    ``a     = (pik_live * sb).sum()``
    ``denom = (n − a) − pik_live * (n − i + 1)``
    ``probs = (1 − sb) * pik_live * ((n − a) − pik_live) / denom``

    One unit is chosen with probability ∝ ``probs``; if
    ``probs.sum() == 0`` (a numerical edge case when a single candidate
    remains) we draw uniformly from the remaining indices.

    The procedure maximises entropy subject to the fixed-size constraint and
    the prescribed first-order probabilities.

    References
    ----------
    Brewer, K. R. W., & Hanif, M. (1983). *Sampling with Unequal
    Probabilities.* Springer-Verlag, New York.
    """
    if not isinstance(pik, np.ndarray):
        pik = np.asarray(pik, dtype=float)

    if np.any(np.isnan(pik)):
        raise ValueError("Missing values detected in the pik vector.")

    live = (pik > eps) & (pik < 1 - eps)
    pik_live = pik[live]
    N = pik_live.size
    if N == 0:
        raise ValueError(
            "All elements in pik are outside the open interval (eps, 1 - eps)."
        )

    n = as_int(np.round(pik_live.sum()))
    rng = rng or np.random.default_rng()

    sb = np.zeros(N, dtype=np.int8)
    for i in range(1, n + 1):
        a = np.dot(pik_live, sb)
        denom = (n - a) - pik_live * (n - i + 1)

        denom = np.where(np.abs(denom) < eps, eps, denom)
        probs = (1 - sb) * pik_live * ((n - a) - pik_live) / denom
        total = probs.sum()
        if total == 0:
            idx = rng.choice(np.flatnonzero(sb == 0))
        else:
            idx = np.searchsorted(np.cumsum(probs / total), rng.random())
        sb[idx] = 1

    out = np.zeros_like(pik, dtype=np.int8)
    out[pik >= 1 - eps] = 1
    out[live] = sb
    return out
