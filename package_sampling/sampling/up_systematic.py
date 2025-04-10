import numpy as np


def up_systematic(pik: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Systematic sampling method (UPsystematic).

    Args:
        pik (np.ndarray): A 1D NumPy array of inclusion probabilities.
        eps (float): Small value for stability in comparison (default: 1e-6).

    Returns:
        np.ndarray: A binary selection array of 0s and 1s.
    """
    if not isinstance(pik, np.ndarray):
        pik = np.array(pik, dtype=np.float64)

    if np.any(np.isnan(pik)):
        raise ValueError("There are missing values in the pik vector")

    s = np.zeros_like(pik, dtype=int)
    mask = (pik > eps) & (pik < 1 - eps)
    pik1 = pik[mask]
    N = len(pik1)

    if N == 0:
        s = (pik > 1 - eps).astype(int)
        return s

    u = np.random.uniform(0, 1)
    a = (np.concatenate(([0], np.cumsum(pik1))) - u) % 1
    s1 = (a[:N] > a[1:]).astype(int)

    s[mask] = s1
    s[pik >= 1 - eps] = 1

    return s
