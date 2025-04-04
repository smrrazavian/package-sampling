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

    if n >= 2:
        pik2 = pik[pik != 1]
        n = np.sum(pik2)
        n = as_int(n)

        piktilde = upme_pik_tilde_from_pik(pik2)
        w = piktilde / (1 - piktilde)

        q = upme_q_from_w(w, n)
        s2 = upme_s_from_q(q)

        s = np.zeros_like(pik, dtype=int)
        s[pik == 1] = 1
        s[pik != 1][s2 == 1] = 1

    elif n == 0:
        s = np.zeros_like(pik, dtype=int)

    elif n == 1:
        s = np.random.multinomial(1, pik)

    return s
