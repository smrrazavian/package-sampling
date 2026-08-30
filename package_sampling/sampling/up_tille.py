"""
Tillé unequal-probability sampling without replacement, fixed size.

Port of R's ``UPtille`` (package *sampling* 2.10).

Reference:
  Deville & Tillé (1998) "Unequal probability sampling without replacement
  through a splitting method", *Biometrika* 85 : 89-101.

The elimination loop calls ``inclusion_probabilities`` once per removal step,
and those calls depend only on ``pik`` — at N=2000 they are ~89% of a draw.
:class:`UPTille` computes the whole sequence once and reuses it (ADR-0002).
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from package_sampling.utils import as_int, check_pik, inclusion_probabilities

from .up_tille_pi2 import up_tille_pi2


class UPTille:
    """
    Tillé unequal-probability sampling design, fixed size.

    Built once from ``pik`` — including the per-step elimination
    probabilities — then drawn from as often as needed.

    Parameters
    ----------
    pik : 1-D array-like of float
        Inclusion probabilities in ``[0, 1]``.
    eps : float, default ``1e-6``
        Threshold separating deterministic units from live ones.

    Attributes
    ----------
    pik : ndarray of float
        The target probabilities, coerced to float.
    n : int
        Sample size over the live units.
    live : ndarray of bool
        Which units are decided at random.
    v : list of ndarray
        One elimination-probability vector per removal step, precomputed from
        ``pik`` alone.  Empty when the design is degenerate.

    Raises
    ------
    ValueError
        If ``pik`` contains NaN, or every unit lies outside
        ``(eps, 1-eps)``.
    """

    #: Whether every draw has the same sample size.
    #: Tille draws exactly ``n`` units, so Sen-Yates-Grundy applies.
    fixed_size = True

    def __init__(
        self,
        pik: Union[List[float], NDArray[np.floating]],
        eps: float = 1e-6,
    ) -> None:
        pik = np.asarray(pik, dtype=float)
        check_pik(pik, nan_message="`pik` contains NaN values.")

        self.pik = pik
        self.eps = eps
        self.live = (pik > eps) & (pik < 1 - eps)
        pik_live = pik[self.live]
        self._n_live_units = pik_live.size
        if self._n_live_units == 0:
            raise ValueError("All inclusion probs are outside ]eps,1-eps[ .")

        self.n = as_int(pik_live.sum())

        self._base = np.zeros_like(pik, dtype=np.int8)
        self._base[pik >= 1 - eps] = 1

        # Elimination probabilities, one vector per removal step. Depends on
        # pik only, so it is built here rather than inside every draw.
        # n == N_live is impossible once as_int rejects non-integer sums
        # (ADR-0007), so there is no census case to skip.
        self.v: list[NDArray[np.floating]] = []
        if 0 < self.n < self._n_live_units:
            prev_b = np.ones(self._n_live_units, dtype=float)
            for k in range(self._n_live_units - self.n):
                a = inclusion_probabilities(pik_live, self._n_live_units - k - 1)
                self.v.append(1.0 - a / prev_b)
                prev_b = a

    def draw(self, rng: np.random.Generator | None = None) -> NDArray[np.int8]:
        """Draw a 0/1 membership mask with exactly :attr:`n` live selections."""
        out = self._base.copy()
        if self.n == 0:
            return out

        rng = rng or np.random.default_rng()
        sb = np.ones(self._n_live_units, dtype=np.int8)
        for v in self.v:
            probs = v * sb
            total = probs.sum()
            if total <= 0:
                idx = rng.choice(np.flatnonzero(sb))
            else:
                idx = np.searchsorted(np.cumsum(probs) / total, rng.random())
            sb[idx] = 0

        out[self.live] = sb
        return out

    def pi2(self) -> NDArray[np.floating]:
        """Joint inclusion probabilities (port of ``UPtillepi2``)."""
        return up_tille_pi2(self.pik, eps=self.eps)


def up_tille(
    pik: Union[List[float], NDArray[np.floating]],
    eps: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int8]:
    """
    Draw a single Tillé sample (one-shot form of :class:`UPTille`).

    Rebuilds the design on every call.  To draw repeatedly from the same
    ``pik``, construct :class:`UPTille` once instead.

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
    return UPTille(pik, eps).draw(rng)
