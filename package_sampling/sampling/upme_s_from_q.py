"""
UPME Sample Selection from Matrix of Probabilities

This module provides an implementation of the sample selection procedure
used in the Unequal Probability Maximum Entropy (UPME) sampling method,
specifically the `UPMEsfromq` algorithm.

Function:
    upme_s_from_q(q: np.ndarray) -> np.ndarray:
        Selects a sample from a matrix `q`, where each entry represents
        the probability of including a unit in the final sample, based on
        backward recursive calculation.

Key characteristics:
- Respects total sample size `n` implicitly encoded in `q`.
- Generates a 0/1 array indicating which items are selected.
- Probabilistic selection ensures reproducibility and randomness.

Example usage:
    >>> q = np.array([[0.1, 0.5], [0.2, 0.7], [0.3, 0.9]])
    >>> upme_s_from_q(q)
    [1, 0, 1]
"""

import numpy as np


def upme_s_from_q(q: np.ndarray) -> np.ndarray:
    """
    Implements the UPMEsfromq sampling method.

    Args:
        q (np.ndarray): A 2D NumPy array where q[k, n] represents the probability of selecting item k.

    Returns:
        np.ndarray: A binary selection array (0 or 1) indicating the selected samples.
    """
    if not isinstance(q, np.ndarray) or q.ndim != 2:
        raise ValueError("Input q must be a 2D NumPy array.")

    N, n = q.shape
    s = np.zeros(N, dtype=int)

    for k in range(N):
        if n > 0 and np.random.uniform(0, 1) < q[k, n - 1]:
            s[k] = 1
            n -= 1

    return s
