"""Validate a vector of first-order inclusion probabilities.

``pi_k = P(k in S)`` is a probability, so it lies in ``[0, 1]``.  A vector with
an entry above 1 is not an inclusion-probability vector at all.

Such vectors arise legitimately: a pips design sets ``pi_k = n * x_k / sum(x)``
from size measures, and a large ``x_k`` pushes ``pi_k`` past 1.  The canonical
remedy is the take-all stratum — cap the offenders at 1, recompute the rest
over the reduced sample size, and iterate — which is exactly what
:func:`package_sampling.utils.inclusion_probabilities` does.

Accepting an out-of-range value instead is not harmless.  The design treats
``pi > 1`` as a certainty unit, so ``E[S_k] = 1`` while the Horvitz-Thompson
estimator still divides by the nominal ``pi_k``.  Unbiasedness requires
``E[S_k] = pi_k``, so the estimator is biased — measured at -7.49% for
``pik = [0.857, 1.0, 1.143]`` (ADR-0008).

R's *sampling* performs no such check; this is a deliberate divergence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


def check_pik(pik: FloatArray, *, nan_message: str) -> None:
    """
    Raise if ``pik`` is not a usable vector of inclusion probabilities.

    Args:
        pik: Already coerced to a float array.
        nan_message: Wording for the missing-value error, so each design can
            keep the message its callers already match on.

    Raises:
        ValueError: If ``pik`` contains NaN, or any entry lies outside
            ``[0, 1]``.
    """
    if np.any(np.isnan(pik)):
        raise ValueError(nan_message)

    bad = (pik < 0.0) | (pik > 1.0)
    if np.any(bad):
        idx = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"inclusion probabilities must lie in [0, 1]; got {pik[idx]!r} "
            f"at index {idx}. Use inclusion_probabilities(x, n) to build a "
            f"valid vector -- it caps take-all units at 1 and redistributes "
            f"the remainder."
        )
