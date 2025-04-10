"""
Module utils.inclusion_probabilities

This module provides a function to compute inclusion probabilities given a
vector of weights and a desired sample size. It handles input validation,
warns of any zeros or negative values, and iteratively adjusts probabilities
to ensure that none exceed 1.
"""

import warnings
from typing import List, Union

import numpy as np


def inclusion_probabilities(a: Union[List[float], np.ndarray], n: int) -> np.ndarray:
    """
    Calculates inclusion probabilities for a given vector of weights.

    Args:
        a (Union[List[float], np.ndarray]): A 1D list or NumPy array representing the weights.
        n (int): The desired sample size.

    Returns:
        np.ndarray: A 1D NumPy array of inclusion probabilities,
            where each entry is between 0 and 1.

    Raises:
        ValueError: If the input vector `a` is empty.
        Warning: If the input vector contains zero or negative values.
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if a.size == 0:
        raise ValueError("Input vector `a` is empty.")

    n_null = np.sum(a == 0)
    n_neg = np.sum(a < 0)
    if n_null > 0:
        warnings.warn("There are zero values in the initial vector `a`.", UserWarning)
    if n_neg > 0:
        warnings.warn(
            f"There are {n_neg} negative value(s) shifted to zero.", UserWarning
        )
    a[a < 0] = 0

    if np.all(a == 0):
        return np.zeros_like(a)

    n = max(0, min(n, np.sum(a > 0)))

    pik1 = n * a / np.sum(a) if n > 0 else np.zeros_like(a, dtype=float)

    positive_mask = pik1 > 0
    if not np.any(positive_mask):
        return pik1

    pik = pik1[positive_mask].copy()
    pik = _adjust_probabilities(pik, n)
    pik1[positive_mask] = pik
    return pik1


def _adjust_probabilities(pik: np.ndarray, n: int, max_iter: int = 100) -> np.ndarray:
    """
    Iteratively adjusts the inclusion probabilities so that none exceed 1.

    Args:
        pik (np.ndarray): The vector of initial inclusion probabilities.
        n (int): The sample size.
        max_iter (int): Maximum iterations to attempt adjustments.

    Returns:
        np.ndarray: The adjusted probabilities.
    """
    iter_count = 0
    prev_l = -1
    while True:
        iter_count += 1
        if iter_count > max_iter:
            break
        is_at_max = pik >= 1
        l = np.sum(is_at_max)
        if l in (0, prev_l):
            break
        prev_l = l

        mask = ~is_at_max
        x = pik[mask]
        total_x = np.sum(x)
        if total_x > 0:
            pik[mask] = (n - l) * (x / total_x)
        else:
            pik[mask] = 0
        pik[is_at_max] = 1
    return pik
