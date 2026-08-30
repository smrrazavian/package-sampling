"""Poisson sampling: each unit is drawn independently with probability pik.

Unlike the fixed-size designs, the realised sample size is random.
Port of R's ``UPpoisson`` (package *sampling* 2.10).
"""

from __future__ import annotations

from typing import List, Union

import numpy as np


class UPPoisson:
    """
    Poisson unequal-probability sampling design.

    Built once from ``pik``, then drawn from as often as needed.

    Parameters
    ----------
    pik : 1-D array-like of float
        Inclusion probabilities in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``pik`` contains NaN.
    """

    #: Whether every draw has the same sample size.
    #: Poisson sample size is RANDOM, so Sen-Yates-Grundy does NOT apply --
    #: it silently returns zero. Use the single-sum variance instead,
    #: sum (1 - pik) * y**2 / pik  (ADR-0005).
    fixed_size = False

    def __init__(self, pik: Union[List[float], np.ndarray]) -> None:
        pik = np.asarray(pik, dtype=float)
        if np.any(np.isnan(pik)):
            raise ValueError("There are missing values in the `pik` vector.")
        self.pik = pik

    def draw(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw a 0/1 membership mask. Sample size is random."""
        rng = rng or np.random.default_rng()
        return np.asarray(rng.random(self.pik.size) < self.pik, dtype=np.int8)

    def pi2(self) -> np.ndarray:
        """
        Joint inclusion probabilities.

        Poisson draws are independent, so ``pi_kl = pi_k * pi_l`` off the
        diagonal and ``pi_kk = pi_k``.  Exact, but **degenerate**: the matrix
        holds nothing the ``pik`` vector does not, since ``pi_kl - pi_k*pi_l``
        is identically zero off the diagonal.  R ships no ``UPpoissonpi2`` and
        that is a judgement, not a gap (ADR-0004).

        Do **not** feed this to a Sen-Yates-Grundy variance: that form assumes
        a fixed sample size and returns zero here (ADR-0005).  Use
        ``sum((1 - pik) * y**2 / pik)`` instead.
        """
        m = np.outer(self.pik, self.pik)
        np.fill_diagonal(m, self.pik)
        return m


def up_poisson(
    pik: Union[List[float], np.ndarray],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Draw a single Poisson sample (one-shot form of :class:`UPPoisson`).

    Parameters
    ----------
    pik : 1-D array-like of float
        Inclusion probabilities in ``[0, 1]``.
    rng : numpy.random.Generator or None
        Random source; a fresh ``default_rng()`` is used if *None*.

    Returns
    -------
    ndarray of int8
        0/1 mask the same shape as ``pik``.

    Raises
    ------
    ValueError
        If ``pik`` contains NaN.
    """
    return UPPoisson(pik).draw(rng)
