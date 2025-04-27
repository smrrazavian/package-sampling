"""
This module implements the maximum entropy sampling method (UPmaxentropy),
used in unequal probability sampling schemes.
"""

import numpy as np

from package_sampling.utils import as_int

from .upme_pik_tilde_from_pik import upme_pik_tilde_from_pik
from .upme_q_from_w import upme_q_from_w
from .upme_s_from_q import upme_s_from_q


def up_max_entropy(pik: np.ndarray) -> np.ndarray:
    """
    Implements the UPmaxentropy sampling method.

    Args:
        pik (np.ndarray): A 1D NumPy array representing inclusion probabilities.

    Returns:
        np.ndarray: A binary selection array (0 or 1) indicating the selected samples.
    """
    if isinstance(pik, (list, tuple)):
        pik = np.array(pik, dtype=np.float64)

    if isinstance(pik, np.ndarray):
        if pik.ndim > 1:
            raise ValueError("pik must be a 1D vector")

    if np.any(pik < 0) or np.any(pik > 1):
        raise ValueError("Inclusion probabilities must be between 0 and 1.")

    n = np.sum(pik)
    n = as_int(n)

    s = np.zeros_like(pik, dtype=np.int64)

    s[pik == 1] = 1

    remaining_mask = pik < 1
    remaining_pik = pik[remaining_mask]
    remaining_n = n - np.sum(s)

    if remaining_n >= 2 and len(remaining_pik) > 0:
        piktilde = upme_pik_tilde_from_pik(remaining_pik)

        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(piktilde < 1, piktilde / (1 - piktilde), np.inf)

        q = upme_q_from_w(w, remaining_n)

        s2 = upme_s_from_q(q)

        s[remaining_mask] = s2

    elif remaining_n == 1 and len(remaining_pik) > 0:
        norm_pik = remaining_pik / np.sum(remaining_pik)
        rng = np.random.default_rng()
        s_remaining = rng.multinomial(1, norm_pik)
        s[remaining_mask] = s_remaining

    return s
