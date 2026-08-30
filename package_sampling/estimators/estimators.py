"""
Design-based estimators for unequal-probability samples.

Ports of R's ``HTestimator``, ``varHT``, ``varest`` and ``Hajekestimator``
(package *sampling* 2.10), none of which had been brought across, plus the
exact design variance the validation battery needs (ADR-0005).

Two families live here and they answer different questions.

*Estimators* — :func:`ht_total`, :func:`hajek_total`, :func:`var_ht`,
:func:`var_syg`, :func:`var_deville` — take the values and probabilities of
the units **in a sample** and estimate a population quantity.

*Design quantities* — :func:`design_variance` — take the whole population and
compute the exact variance of the Horvitz-Thompson estimator under a design.
Use this to check a design, not to analyse a survey.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


def _as_pair(y: Any, pik: Any) -> tuple[FloatArray, FloatArray]:
    y = np.asarray(y, dtype=float)
    pik = np.asarray(pik, dtype=float)
    if y.shape != pik.shape:
        raise ValueError(
            f"`y` and `pik` must have the same length; got {y.shape} "
            f"and {pik.shape}."
        )
    if np.any(np.isnan(y)) or np.any(np.isnan(pik)):
        raise ValueError("`y` and `pik` must not contain NaN.")
    if np.any(pik <= 0):
        raise ValueError(
            "every sampled unit needs pik > 0; a unit that cannot be drawn "
            "carries no Horvitz-Thompson weight."
        )
    return y, pik


def _delta(pi2: Any) -> tuple[FloatArray, FloatArray]:
    """Return ``(pik, Delta)`` where ``Delta = pi2 - outer(pik, pik)``."""
    pi2 = np.asarray(pi2, dtype=float)
    if pi2.ndim != 2 or pi2.shape[0] != pi2.shape[1]:
        raise ValueError(f"`pi2` must be a square matrix; got {pi2.shape}.")
    pik = np.diag(pi2).copy()
    delta = pi2 - np.outer(pik, pik)
    np.fill_diagonal(delta, pik * (1.0 - pik))
    return pik, delta


def ht_total(y: Any, pik: Any) -> float:
    """
    Horvitz-Thompson estimator of a population total (R ``HTestimator``).

    Args:
        y: Values of the sampled units.
        pik: Their first-order inclusion probabilities.

    Returns:
        float: ``sum(y / pik)``, unbiased for the population total whenever
        every unit has ``pik > 0``.
    """
    y, pik = _as_pair(y, pik)
    return float(np.sum(y / pik))


def hajek_total(y: Any, pik: Any, N: int | None = None) -> float:
    """
    Hajek estimator (R ``Hajekestimator``).

    Divides by the *estimated* population size rather than the known one,
    which trades unbiasedness for stability when the realised sample size
    varies.

    Args:
        y: Values of the sampled units.
        pik: Their first-order inclusion probabilities.
        N: Population size.  If omitted the mean is returned, matching R's
            default.

    Returns:
        float: The estimated total when ``N`` is given, else the mean.
    """
    y, pik = _as_pair(y, pik)
    mean = float(np.sum(y / pik) / np.sum(1.0 / pik))
    return mean if N is None else float(N) * mean


def var_ht(y: Any, pi2: Any) -> float:
    """
    Horvitz-Thompson variance estimator (R ``varHT``, ``method = 1``).

    Args:
        y: Values of the sampled units.
        pi2: Their joint-inclusion submatrix, with first-order probabilities
            on the diagonal.

    Returns:
        float: ``sum(y_k y_l Delta_kl / (pi_k pi_l pi_kl))``.

    Raises:
        ValueError: If ``pi2`` is not square or does not match ``y``.
    """
    pik, delta = _delta(pi2)
    y, pik = _as_pair(y, pik)
    pi2 = np.asarray(pi2, dtype=float)
    return float(np.sum(np.outer(y, y) * delta / (np.outer(pik, pik) * pi2)))


def var_syg(y: Any, pi2: Any) -> float:
    """
    Sen-Yates-Grundy variance estimator (R ``varHT``, ``method = 2``).

    **Fixed-size designs only.**  The derivation treats ``sum(S_k) = n`` as a
    constant, so for a design with a random sample size every off-diagonal
    ``Delta_kl`` is zero and the formula returns 0 — a plausible number that
    is completely wrong.  Poisson sampling hits exactly that, so the case is
    rejected rather than answered (ADR-0005).

    Args:
        y: Values of the sampled units.
        pi2: Their joint-inclusion submatrix.

    Returns:
        float: ``0.5 * sum((y_k/pi_k - y_l/pi_l)^2 (pi_k pi_l - pi_kl)/pi_kl)``.

    Raises:
        ValueError: If ``pi2`` describes a design with independent draws.
    """
    pik, delta = _delta(pi2)
    y, pik = _as_pair(y, pik)
    pi2 = np.asarray(pi2, dtype=float)

    off = ~np.eye(pik.size, dtype=bool)
    if pik.size > 1 and np.allclose(delta[off], 0.0, atol=1e-12):
        raise ValueError(
            "Sen-Yates-Grundy is defined for fixed-size designs only. Every "
            "off-diagonal pi_kl equals pi_k * pi_l here, so the draws are "
            "independent and this formula would return 0. Use var_ht, or "
            "design_variance for the exact value."
        )

    diff2 = np.subtract.outer(y / pik, y / pik) ** 2
    return float(0.5 * np.sum(diff2 * (np.outer(pik, pik) - pi2) / pi2))


def var_deville(y: Any, pik: Any) -> float:
    """
    Deville's variance approximation (R ``varest``).

    Needs only first-order probabilities — no joint-inclusion matrix — which
    is why Matei & Tille (2005) found it accurate enough to make pi2
    unnecessary for variance estimation under high-entropy designs.

    Args:
        y: Values of the sampled units.
        pik: Their first-order inclusion probabilities.

    Returns:
        float: The approximate variance of the Horvitz-Thompson total.
    """
    # ponytail: R's varest also takes calibration variables Xs via MASS::ginv;
    # not ported until something needs it.
    y, pik = _as_pair(y, pik)
    a = (1.0 - pik) / np.sum(1.0 - pik)
    centre = np.sum(a * y / pik)
    return float(
        np.sum((1.0 - pik) * (y / pik - centre) ** 2) / (1.0 - np.sum(a**2))
    )


def design_variance(y: Any, pik: Any, pi2: Any) -> float:
    """
    Exact variance of the Horvitz-Thompson total under a design.

    A population quantity, not an estimate from a sample: it needs every
    unit's value.  Valid for **any** design, fixed-size or not, because it
    makes no assumption about the sample size (ADR-0005).

    Args:
        y: Values of every population unit.
        pik: Their first-order inclusion probabilities.
        pi2: The full joint-inclusion matrix.

    Returns:
        float: ``sum_k sum_l Delta_kl y_k y_l / (pi_k pi_l)``.
    """
    y, pik = _as_pair(y, pik)
    _, delta = _delta(pi2)
    return float(np.sum(np.outer(y, y) * delta / np.outer(pik, pik)))
