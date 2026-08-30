"""
Maximum-entropy sampling design (port of R's ``UPmaxentropy``).

Draws a fixed-size sample that matches the target inclusion probabilities and
maximises Shannon entropy subject to that constraint, via the sequential
machinery π̃ → w → q → s (Tillé & Matei, 2016, ch. 11).

Building the design from ``pik`` is expensive and drawing from it is not: at
N=2000 the π̃ fixed point and the q matrix cost ~1.9 s between them while a draw
costs ~2 ms.  Both are functions of ``pik`` alone, so they are computed once in
:class:`UPMaxEntropy` and reused across draws (ADR-0002).

Deterministic units: π ≥ 1−eps is always selected, π ≤ eps never is.  Only the
live units drive the entropy loop.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from package_sampling.utils import as_int, check_pik

from .up_max_entropy_pi2 import up_max_entropy_pi2
from .upme_pik_tilde_from_pik import upme_pik_tilde_from_pik
from .upme_q_from_w import upme_q_from_w
from .upme_s_from_q import upme_s_from_q


class UPMaxEntropy:
    """
    Maximum-entropy unequal-probability sampling design, fixed size.

    Built once from ``pik`` — including the π̃ fixed point and the q matrix —
    then drawn from as often as needed.

    Parameters
    ----------
    pik : 1-D array-like of float
        Target inclusion probabilities in ``[0, 1]``.
    eps : float, default ``1e-6``
        Threshold separating deterministic units from live ones.

    Attributes
    ----------
    pik : ndarray of float
        The target probabilities, coerced to float.
    n : int
        Sample size, ``round(sum(pik))``.  Every draw has exactly this many
        selected units.
    live : ndarray of bool
        Which units are decided at random.
    q : ndarray of float or None
        The sequential conditional-probability matrix, or *None* when the
        design is degenerate enough not to need one (n ≤ 1, or every unit
        already deterministic).

    Raises
    ------
    ValueError
        If ``pik`` is not 1-D, or contains NaN or values outside ``[0, 1]``.
    """

    #: Whether every draw has the same sample size.
    #: Max-entropy draws exactly ``n`` units, so Sen-Yates-Grundy applies.
    fixed_size = True

    def __init__(
        self,
        pik: Union[List[float], NDArray[np.floating]],
        eps: float = 1e-6,
    ) -> None:
        pik = np.asarray(pik, dtype=float)
        if pik.ndim != 1:
            raise ValueError("pik must be a 1-D vector")
        check_pik(
            pik, nan_message="Inclusion probabilities must be between 0 and 1."
        )

        self.pik = pik
        self.eps = eps
        self.live = (pik > eps) & (pik < 1 - eps)

        self._base = np.zeros_like(pik, dtype=np.int8)
        self._base[pik >= 1 - eps] = 1
        self._n_certain = int(self._base.sum())

        # n comes from the LIVE units plus the certain ones, never from the
        # raw total: units below eps contribute a sliver that is excluded from
        # the design anyway, and letting it reach as_int would reject a design
        # that is perfectly well defined (ADR-0007).
        self._n_live = as_int(pik[self.live].sum())
        self.n = self._n_certain + self._n_live

        self.q: NDArray[np.floating] | None = None
        if self.n >= 2 and self._n_live > 0:
            pik_tilde = upme_pik_tilde_from_pik(pik[self.live], eps=eps)
            with np.errstate(divide="ignore"):
                w = pik_tilde / (1.0 - pik_tilde)
            self.q = upme_q_from_w(w, self._n_live)

    def draw(self, rng: np.random.Generator | None = None) -> NDArray[np.int8]:
        """Draw a 0/1 membership mask with exactly :attr:`n` selected units."""
        sel = self._base.copy()
        if self.n == 0 or self.n == self._n_certain:
            return sel

        rng = rng or np.random.default_rng()

        if self.n == 1:
            sel[rng.choice(self.pik.size, p=self.pik / self.pik.sum())] = 1
            return sel

        sel[self.live] = upme_s_from_q(self.q, rng=rng).astype(np.int8)
        return sel

    def pi2(self) -> NDArray[np.floating]:
        """
        Joint inclusion probabilities (port of ``UPmaxentropypi2``).

        Accurate to roughly ``1e-8``, not to machine precision: the matrix is
        built from the pi-tilde fixed point, which stops at ``eps``.  The
        row-sum identity ``sum_l pi_kl = n * pi_k`` therefore holds only to
        that tolerance, where UPTille and UPSystematic satisfy it to an ulp.
        """
        return up_max_entropy_pi2(self.pik, eps=self.eps)


def up_max_entropy(
    pik: Union[List[float], NDArray[np.floating]],
    eps: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int8]:
    """
    Draw a single max-entropy sample (one-shot form of :class:`UPMaxEntropy`).

    Rebuilds the design on every call.  To draw repeatedly from the same
    ``pik``, construct :class:`UPMaxEntropy` once instead.

    Parameters
    ----------
    pik : 1-D array-like of float
        Target inclusion probabilities in ``[0, 1]``.
    eps : float, default ``1e-6``
        Threshold separating deterministic units from live ones.
    rng : numpy.random.Generator or None
        Random source; a fresh ``default_rng()`` is used if *None*.

    Returns
    -------
    ndarray of int8
        0/1 mask with ``sum() == round(sum(pik))``.
    """
    return UPMaxEntropy(pik, eps).draw(rng)
