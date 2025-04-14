"""
UPME Adjusted Inclusion Probabilities (Tilde Version)

This module implements an iterative method for adjusting inclusion probabilities
in unequal probability sampling. The procedure is part of the Maximum Entropy
sampling design (UPME), where adjusted inclusion probabilities are derived
such that they are consistent with the maximum entropy criterion.

Function:
    upme_pik_tilde_from_pik(pik, eps=1e-6) -> np.ndarray:
        Iteratively computes a modified version of the inclusion probabilities (`pik`)
        to achieve balance in expected sample size and probabilistic consistency.

This method ensures:
- Probabilities remain within the [0, 1] interval.
- Total inclusion expectation is preserved.
- Convergence is controlled via a specified threshold (`eps`).

Dependencies:
- `upme_q_from_w`: Computes transition matrix `q` from importance weights.
- `upme_pik_from_q`: Computes adjusted probabilities from matrix `q`.
- `as_int`: Safely casts sums to integers while handling rounding checks.

Typical usage example:
    >>> pik = [0.2, 0.4, 0.6]
    >>> upme_pik_tilde_from_pik(pik)
    [0.19721907 0.35431553 0.6484654 ]
"""

import warnings
from typing import List, Union

import numpy as np

from package_sampling.utils import as_int

from .upme_pik_from_q import upme_pik_from_q
from .upme_q_from_w import upme_q_from_w


def upme_pik_tilde_from_pik(
    pik: Union[List[float], np.ndarray], eps: float = 1e-6
) -> np.ndarray:
    """
    Computes the adjusted inclusion probabilities using an iterative method.

    Args:
        pik (Union[List[float], np.ndarray]): A 1D list or NumPy array representing inclusion probabilities.
        eps (float, optional): A small threshold for convergence criteria. Defaults to 1e-6.

    Returns:
        np.ndarray: A 1D array of the same shape as `pik`, containing adjusted inclusion probabilities.
    """
    if not isinstance(pik, np.ndarray):
        pik = np.array(pik, dtype=np.float64)

    if len(pik) == 0:
        return np.zeros_like(pik)

    if np.any(pik < 0) or np.any(pik > 1):
        warnings.warn("Clipping input probabilities to [0,1].", UserWarning)
        pik = np.clip(pik, 0, 1)

    if np.all(pik == 1):
        return np.ones_like(pik)

    n = as_int(np.sum(pik))
    if n == 0:
        return np.zeros_like(pik)

    pikt = pik.copy()
    max_iter = 1000
    iterations = 0
    arr = 1.0

    while arr > eps and iterations < max_iter:
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(pikt < 1, pikt / (1 - pikt), np.inf)

        if np.all(w == np.inf):
            return np.ones_like(pik)

        q = upme_q_from_w(w, n)

        pik_from_q = upme_pik_from_q(q)

        pik_from_q = np.nan_to_num(pik_from_q, nan=0.0)

        pikt1 = np.where(
            np.isnan(pik_from_q), pikt, np.clip(pikt + pik - pik_from_q, 0, 1)
        )
        pikt1[np.isclose(pik, 1)] = 1

        if np.sum(pikt1) > 0:
            scale_factor = np.sum(pik) / np.sum(pikt1)
            pikt1 *= scale_factor

        arr = np.sum(np.abs(pikt - pikt1))

        if np.allclose(pikt, pikt1, atol=eps):
            break

        pikt = pikt1
        iterations += 1

    if iterations == max_iter:
        print("Warning: Max iterations reached, may not have converged.")

    return pikt
