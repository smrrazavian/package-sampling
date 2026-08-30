"""
Systematic sampling with unequal probabilities.

Port of R's ``UPsystematic`` / ``UPsystematicpi2`` (package *sampling* 2.10).

Units with π outside ``(eps, 1-eps)`` are deterministic: π ≥ 1−eps is always
selected, π ≤ eps never is.  Only the live units drive the systematic sweep.

Note the π₂ block uses R's own thresholds ``0 < π < 1`` rather than the eps
window — that is parity with ``UPsystematicpi2``, not drift.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from package_sampling.utils import as_int, check_pik

from .up_systematic_pi2 import up_systematic_pi2


class UPSystematic:
    """
    Systematic unequal-probability sampling design.

    Built once from ``pik``, then drawn from as often as needed.

    Parameters
    ----------
    pik : 1-D array-like of float
        Inclusion probabilities in ``[0, 1]``.
    eps : float, default ``1e-6``
        Threshold separating deterministic units from live ones.

    Attributes
    ----------
    n : int
        Realised sample size: every draw selects exactly this many units,
        certain units included.

    Raises
    ------
    ValueError
        If ``pik`` contains NaN.
    """

    #: Whether every draw has the same sample size.
    #: Systematic draws a fixed number of units, so Sen-Yates-Grundy applies
    #: -- though it violates the Yates-Grundy condition, so estimates may
    #: come out negative.
    fixed_size = True

    def __init__(
        self,
        pik: Union[List[float], np.ndarray],
        eps: float = 1e-6,
    ) -> None:
        pik = np.asarray(pik, dtype=float)
        check_pik(pik, nan_message="There are missing values in the pik vector")

        self.pik = pik
        self.eps = eps
        self.live = (pik > eps) & (pik < 1 - eps)
        self._certain = pik >= 1 - eps

        # Systematic is fixed-size only when sum(pik_live) is a whole number:
        # at 24.6 the realised size oscillates between 24 and 25, so
        # fixed_size would be a false claim. as_int enforces it (ADR-0007).
        self._n_live = as_int(pik[self.live].sum())
        self.n = int(self._certain.sum()) + self._n_live

    def draw(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw a 0/1 membership mask."""
        s = np.zeros_like(self.pik, dtype=np.int8)
        s[self._certain] = 1

        pik1 = self.pik[self.live]
        n_live = pik1.size
        if n_live:
            rng = rng or np.random.default_rng()
            u = rng.random()
            a = (np.concatenate(([0.0], np.cumsum(pik1))) - u) % 1
            s[self.live] = (a[:n_live] > a[1:]).astype(np.int8)

        return s

    def pi2(self) -> np.ndarray:
        """Joint inclusion probabilities (port of ``UPsystematicpi2``)."""
        return up_systematic_pi2(self.pik)


def up_systematic(
    pik: Union[List[float], np.ndarray],
    eps: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Draw a single systematic sample (one-shot form of :class:`UPSystematic`).

    Parameters
    ----------
    pik : 1-D array-like of float
        Inclusion probabilities in ``[0, 1]``.
    eps : float, default ``1e-6``
        Threshold separating deterministic units from live ones.
    rng : numpy.random.Generator or None
        Random source; a fresh ``default_rng()`` is used if *None*.

    Returns
    -------
    ndarray of int8
        0/1 mask the same shape as ``pik``.
    """
    return UPSystematic(pik, eps).draw(rng)
