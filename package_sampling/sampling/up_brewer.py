"""
Brewer unequal-probability sampling without replacement, fixed size.

Port of R's ``UPbrewer`` (package *sampling* 2.10), vectorised.

Reference: Brewer & Hanif (1983), *Sampling with Unequal Probabilities*,
Springer-Verlag, New York (§8.5).

Unlike the other designs, Brewer has nothing worth precomputing: every step of
the take-one loop reads the running selection vector, so the whole draw depends
on the random stream.  :class:`UPBrewer` exists for a uniform design interface,
not for speed.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from package_sampling.utils import as_int


class UPBrewer:
    """
    Brewer unequal-probability sampling design, fixed size.

    Implements the Brewer–Hanif algorithm.  All live units
    (``eps < πᵢ < 1−eps``) go through a sequential take-one loop; units with
    π ≈ 0 or π ≈ 1 are settled deterministically.

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

    Raises
    ------
    ValueError
        If ``pik`` contains NaN, or every unit lies outside
        ``(eps, 1-eps)``.
    """

    #: Whether every draw has the same sample size.
    #: Brewer draws exactly ``n`` units, so Sen-Yates-Grundy applies.
    fixed_size = True

    def __init__(
        self,
        pik: Union[List[float], NDArray[np.floating]],
        eps: float = 1e-6,
    ) -> None:
        pik = np.asarray(pik, dtype=float)
        if np.any(np.isnan(pik)):
            raise ValueError("Missing values detected in the pik vector.")

        self.pik = pik
        self.eps = eps
        self.live = (pik > eps) & (pik < 1 - eps)
        self._pik_live = pik[self.live]
        if self._pik_live.size == 0:
            raise ValueError(
                "All elements in pik are outside the open interval "
                "(eps, 1 - eps)."
            )

        self.n = as_int(np.round(self._pik_live.sum()))

        self._base = np.zeros_like(pik, dtype=np.int8)
        self._base[pik >= 1 - eps] = 1

    def draw(self, rng: np.random.Generator | None = None) -> NDArray[np.int8]:
        """Draw a 0/1 membership mask with exactly :attr:`n` live selections."""
        rng = rng or np.random.default_rng()
        pik_live = self._pik_live
        eps = self.eps
        n = self.n

        sb = np.zeros(pik_live.size, dtype=np.int8)
        for i in range(1, n + 1):
            a = np.dot(pik_live, sb)
            denom = (n - a) - pik_live * (n - i + 1)
            denom = np.where(np.abs(denom) < eps, eps, denom)
            probs = (1 - sb) * pik_live * ((n - a) - pik_live) / denom
            total = probs.sum()
            if total == 0:
                idx = rng.choice(np.flatnonzero(sb == 0))
            else:
                idx = np.searchsorted(np.cumsum(probs / total), rng.random())
            sb[idx] = 1

        out = self._base.copy()
        out[self.live] = sb
        return out

    def pi2(self) -> NDArray[np.floating]:
        """
        Not available.

        Brewer's joint inclusion probabilities have no known closed form,
        which is why R's *sampling* ships no ``UPbrewerpi2`` either
        (ADR-0004).  Estimate them by Monte Carlo if you need them.
        """
        raise NotImplementedError(
            "Brewer's joint inclusion probabilities have no known closed "
            "form; R's sampling omits UPbrewerpi2 for the same reason. "
            "Estimate them by Monte Carlo instead."
        )


def up_brewer(
    pik: Union[List[float], NDArray[np.floating]],
    eps: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int8]:
    """
    Draw a single Brewer sample (one-shot form of :class:`UPBrewer`).

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
        0/1 mask the same shape as ``pik``, with ``round(sum(pik_live))``
        selected live units.
    """
    return UPBrewer(pik, eps).draw(rng)
