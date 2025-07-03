"""
UPsystematic Sampling Method

This module provides an implementation of the Unequal Probability (UP) systematic
sampling method, commonly used in survey sampling when units have unequal
inclusion probabilities.

Function:
    up_systematic(pik: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        Applies systematic sampling to a vector of inclusion probabilities and
        returns a binary selection array.

The implementation adheres to probabilistic principles for selecting a sample
according to the specified inclusion probabilities. It is robust to edge cases
such as missing values and handles deterministic inclusions when probabilities
are close to 0 or 1.

Example usage:
    >>> pik = np.array([0.2, 0.3, 0.5])
    >>> up_systematic(pik)
    array([0, 0, 1])
"""

import numpy as np


def up_systematic(
    pik: np.ndarray,
    eps: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Systematic sampling method (UPsystematic).

    Args:
        pik (np.ndarray): A 1D NumPy array of inclusion probabilities.
        eps (float): Small value for stability in comparison (default: 1e-6).

    Returns:
        np.ndarray: A binary selection array of 0s and 1s.
    """
    if not isinstance(pik, np.ndarray):
        pik = np.array(pik, dtype=float)

    if np.any(np.isnan(pik)):
        raise ValueError("There are missing values in the pik vector")

    s = np.zeros_like(pik, dtype=np.int8)
    s[pik >= 1 - eps] = 1

    mask = (pik > eps) & (pik < 1 - eps)
    pik1 = pik[mask]
    N = len(pik1)

    if N > 0:
        rng = rng or np.random.default_rng()
        u = rng.random()
        a = (np.concatenate(([0], np.cumsum(pik1))) - u) % 1
        s1 = (a[:N] > a[1:]).astype(np.int8)
        s[mask] = s1

    return s
