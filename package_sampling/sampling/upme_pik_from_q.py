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

    N, n = q.shape
    pro = np.zeros((N, n))

    pro[0, -1] = 1

    pro[1:, 1:] += pro[:-1, 1:] * (1 - q[:-1, 1:])
    pro[1:, :-1] += pro[:-1, 1:] * q[:-1, 1:]

    pro[1:, 0] += pro[:-1, 0] * (1 - q[:-1, 0])

    return np.sum(pro * q, axis=1)
