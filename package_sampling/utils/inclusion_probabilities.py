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
        np.ndarray: A 1D NumPy array of inclusion probabilities, where each entry is between 0 and 1.

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

    pik1 = np.zeros_like(a, dtype=float)
    if n > 0:
        pik1 = n * a / np.sum(a)

    positive_mask = pik1 > 0
    if not np.any(positive_mask):
        return pik1

    pik = pik1[positive_mask].copy()

    max_iter = 100
    iter_count = 0
    prev_l = -1

    while True:
        iter_count += 1
        if iter_count > max_iter:
            break

        list_ge_1 = pik >= 1
        l = np.sum(list_ge_1)

        if l == 0 or l == prev_l:
            break

        prev_l = l
        x = pik[~list_ge_1]

        if x.size > 0 and np.sum(x) > 0:
            x = x / np.sum(x)
            pik[~list_ge_1] = (n - l) * x
        else:
            idx = np.where(~list_ge_1)[0]
            if idx.size > 0:
                pik[~list_ge_1] = 0

        pik[list_ge_1] = 1

    pik1[positive_mask] = pik
    return pik1
