"""Computes the inclusion probabilities from a given probability matrix q.

Mathematics of the UPMEpikfromq:
- Let q be the probability matrix, where q(i, j) represents the probability of selecting the j-th element from the i-th population.
- The algorithm computes the inclusion probabilities by applying the given probability matrix q to the inclusion probabilities of the population.
- The inclusion probabilities are computed iteratively using the given probability matrix q.
- The final inclusion probabilities are returned as a 1D array.
"""

import numpy as np


def upme_pik_from_q(q: np.ndarray) -> np.ndarray:
    """
    Compute the inclusion probabilities from q.

    Parameters:
        q (np.ndarray): A 2D numpy array representing the probability matrix.

    Returns:
        np.ndarray: A 1D array representing the row sums of the computed probabilities.
    """
    if isinstance(q, list):
        q = np.array(q)

    if not isinstance(q, np.ndarray) or q.ndim != 2:
        raise ValueError("Input q must be a 2D matrix (NumPy array or list).")

    if np.all(q == 1.0):
        return np.ones(q.shape[0])

    N, n = q.shape
    pro = np.zeros((N, n))

    if n > 0:
        pro[0, n - 1] = 1

    for i in range(1, N):
        for j in range(1, n):
            pro[i, j] += pro[i - 1, j] * (1 - q[i - 1, j])
            pro[i, j - 1] += pro[i - 1, j] * q[i - 1, j]

    for i in range(1, N):
        pro[i, 0] += pro[i - 1, 0] * (1 - q[i - 1, 0])

    result = np.zeros(N)
    for i in range(N):
        for j in range(n):
            result[i] += pro[i, j] * q[i, j]

    return result
