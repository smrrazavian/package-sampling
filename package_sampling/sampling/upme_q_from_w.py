"""
Computes the q matrix from the given w vector for Unequal Probability (UP) sampling.

This function constructs an inclusion probability matrix `q` based on the
provided weight vector `w` and sample size `n`. It utilizes recursive
probability calculations.
"""

import numpy as np


def upme_q_from_w(w, n: int) -> np.ndarray:
    """
    Computes the probability matrix `q` from a given weight vector `w`.

    Args:
        w (Union[np.ndarray, list]): A 1D weight vector (NumPy array or Python list).
        n (int): The sample size.

    Returns:
        np.ndarray: A 2D array (N x n) representing the probability matrix `q`.

    Raises:
        ValueError: If `w` is not a valid 1D array or list.
        ValueError: If `n` is greater than the length of `w`.
    """

    if isinstance(w, list):
        w = np.array(w, dtype=np.float64)

    if not isinstance(w, np.ndarray) or w.ndim != 1:
        raise ValueError("Input w must be a 1D NumPy array or list.")
    if n > len(w):
        raise ValueError("Sample size n cannot be larger than the length of w.")

    N = len(w)

    if np.all(w == 0):
        return np.zeros((N, n))

    expa = np.zeros((N, n))
    q = np.zeros((N, n))

    expa[:, 0] = np.cumsum(w[::-1])[::-1]

    for i in range(N - n + 1, N):
        if np.any(w[i:N] == 0):
            expa[i, N - i - 1] = 0
        else:
            expa[i, N - i - 1] = np.exp(np.sum(np.log(w[i:N])))

    for i in range(N - 2, -1, -1):
        for z in range(1, min(N - i, n)):
            expa[i, z] = w[i] * expa[i + 1, z - 1] + expa[i + 1, z]

    valid_expa = np.where(expa[:, 0] == 0, np.inf, expa[:, 0])
    q[:, 0] = w / valid_expa

    for i in range(N - n + 1, N):
        q[i, N - i - 1] = 1

    for i in range(N - 2, -1, -1):
        for z in range(1, min(N - i, n)):
            if expa[i, z] == 0:
                q[i, z] = 0
            else:
                q[i, z] = w[i] * expa[i + 1, z - 1] / expa[i, z]

    return q
