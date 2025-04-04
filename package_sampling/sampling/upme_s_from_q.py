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
