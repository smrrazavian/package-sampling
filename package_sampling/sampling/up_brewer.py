"""
Implements Brewer's method for Unequal Probability (UP) sampling.

This algorithm selects a fixed number of elements from a population
while maintaining the predefined inclusion probabilities.

Mathematics of the UPbrewer:
- Let pik be the inclusion probability of the i-th element.
- The algorithm selects a sample of size n from the population.
- For each element, the algorithm selects it with probability pik.
- If the selection vector `sb` contains more than n `1`s, it means that some elements were selected more than once.
- In this case, the algorithm reduces the probability of selecting the element by multiplying the inclusion probability by `1 - (1/n)`.
- The algorithm continues until the selection vector `sb` contains exactly n `1`s.
- The final selection vector `sb` contains the selected elements.

"""

import numpy as np
from package_sampling.utils import as_int


def up_brewer(pik: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Selects a sample using Brewer's method for Unequal Probability sampling.

    Args:
        pik (np.ndarray): A 1D NumPy array representing inclusion probabilities of each element.
            Values should be in the range (0,1).
        eps (float, optional): A small threshold to handle floating-point precision issues. Defaults to 1e-6.

    Returns:
        np.ndarray: A 1D array of the same shape as `pik`, containing only 0s and 1s.
        A `1` indicates that the corresponding element was selected in the sample.

    Raises:
        ValueError: If `pik` contains NaN values.
        ValueError: If all elements of `pik` are outside the range (eps, 1 - eps).
        ZeroDivisionError: If a division by zero occurs in the selection process.

    Notes:
        - This method ensures that the final selection matches the expected sample size.
        - The function modifies a selection vector (`sb`) iteratively to ensure a valid sample.
        - The probability update step uses a cumulative probability approach.
    """

    if np.any(np.isnan(pik)):
        raise ValueError("Missing values detected in the pik vector.")

    eligible_mask = (pik > eps) & (pik < 1 - eps)
    pikb = pik[eligible_mask]

    if pikb.size == 0:
        raise ValueError("All elements in pik are outside the range [eps, 1-eps].")

    result = np.copy(pik)
    sb = np.zeros_like(pikb, dtype=int)
    total_pikb = as_int(np.sum(pikb))

    for i in range(total_pikb):
        a = np.dot(pikb, sb)
        denom = (total_pikb - a) - pikb * (total_pikb - i + 1)

        denom = np.where(np.abs(denom) < eps, eps, denom)

        p = (1 - sb) * pikb * ((total_pikb - a) - pikb) / denom
        p_sum = np.sum(p)

        if p_sum == 0:
            continue

        p /= p_sum
        cumulative_p = np.cumsum(p)
        j = np.searchsorted(cumulative_p, np.random.uniform())
        sb[j] = 1

    result[eligible_mask] = sb
    return result
