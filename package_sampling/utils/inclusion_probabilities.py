"""
Module utils.inclusion_probabilities

Compute inclusion probabilities for a weight vector and a target sample
size. Handles validation, warns about zeros/negatives, and iteratively
adjusts probabilities so none exceed one.
"""

import warnings
from typing import Any, List, Union, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


def inclusion_probabilities(
    a: Union[List[float], np.ndarray],
    n: int,
) -> FloatArray:
    """
    Calculates inclusion probabilities for a given vector of weights.

    Args:
        a (Union[List[float], np.ndarray]):
            1-D list or NumPy array representing the weights.
        n (int): The desired sample size.

    Returns:
        FloatArray:
            Inclusion probabilities, each between 0 and 1.

    Raises:
        ValueError: If the input vector `a` is empty.
        Warning: If the input vector contains zero or negative values.
    """
    arr = cast(FloatArray, np.asarray(a, dtype=float))
    if arr.size == 0:
        raise ValueError("Input vector `a` is empty.")

    n_null = int(np.sum(arr == 0))
    n_neg = int(np.sum(arr < 0))
    if n_null > 0:
        warnings.warn(
            "There are zero values in the initial vector `a`.",
            UserWarning,
        )
    if n_neg > 0:
        warnings.warn(
            f"There are {n_neg} negative value(s) shifted to zero.",
            UserWarning,
        )
        arr[arr < 0] = 0

    if np.all(arr == 0):
        return cast(FloatArray, np.zeros_like(arr))

    live_count = int(np.sum(arr > 0))
    n = max(0, min(n, live_count))

    pik1: FloatArray
    if n > 0:
        pik1 = cast(FloatArray, n * arr / np.sum(arr))
    else:
        pik1 = cast(FloatArray, np.zeros_like(arr, dtype=float))

    positive_mask = pik1 > 0
    if not np.any(positive_mask):
        return pik1

    pik = cast(FloatArray, pik1[positive_mask].copy())
    pik = _adjust_probabilities(pik, n)
    pik1[positive_mask] = pik
    return pik1


def _adjust_probabilities(
    pik: FloatArray,
    n: int,
    max_iter: int = 100,
) -> FloatArray:
    """
    Iteratively adjusts the inclusion probabilities so that none exceed 1.

    Args:
        pik (FloatArray): The vector of initial inclusion probabilities.
        n (int): The sample size.
        max_iter (int): Maximum iterations to attempt adjustments.

    Returns:
        FloatArray: The adjusted probabilities.
    """
    iter_count = 0
    prev_maxed = -1
    while True:
        iter_count += 1
        if iter_count > max_iter:
            break
        is_at_max = pik >= 1
        maxed_count = int(np.sum(is_at_max))
        if maxed_count in (0, prev_maxed):
            break
        prev_maxed = maxed_count

        mask = ~is_at_max
        x = pik[mask]
        total_x = np.sum(x)
        if total_x > 0:
            pik[mask] = (n - maxed_count) * (x / total_x)
        else:
            pik[mask] = 0
        pik[is_at_max] = 1.0
    return pik
