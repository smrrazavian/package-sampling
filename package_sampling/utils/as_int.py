"""Coerce a float sample size to an int, rejecting genuinely fractional input.

R's ``.as_int`` warns and returns the value **unrounded**, so a fractional
sample size flows on into loop bounds — ``UPtillepi2(c(0.9, 0.9))`` ends up
running ``1:(2 - 1.8)``, which R evaluates as a single iteration.  That result
is an artifact of R's range operator, not a design choice, so we reject the
input instead (ADR-0007).
"""

import numpy as np

#: sqrt(machine epsilon), the tolerance R's all.equal uses by default.
_TOLERANCE = float(np.sqrt(np.finfo(float).eps))


def as_int(x: float) -> int:
    """
    Convert a number to an int, allowing only floating-point noise.

    Args:
        x (float): The number to convert.

    Returns:
        int: The integer value of `x`.

    Raises:
        ValueError: If `x` is not within 1e-6 of an integer, or lies outside
            the 32-bit integer range.
    """
    int_min, int_max = np.iinfo(np.int32).min, np.iinfo(np.int32).max

    if isinstance(x, (int, np.integer)):
        return int(x)

    if x < int_min or x > int_max:
        raise ValueError(
            f"The input {x} exceeds the allowable integer range "
            f"[{int_min}, {int_max}]."
        )

    rounded_x = round(x)

    # R's .as_int defers to all.equal, whose tolerance is sqrt(machine eps)
    # and which compares relatively unless the target is ~0. Matching it keeps
    # the accept/reject boundary identical to the package we port (ADR-0007).
    # It is ~7 orders of magnitude above the measured error in sum(pik):
    # NumPy sums pairwise, bounded by (log2 N + 1) * u * sum|x|, which stays
    # near 2e-16 relative even at N = 1e6.
    tol = _TOLERANCE * abs(rounded_x) if rounded_x else _TOLERANCE

    if abs(rounded_x - x) > tol:
        raise ValueError(
            f"{x} is not an integer. A fixed-size design needs an integer "
            f"sample size, so sum(pik) over the live units must be a whole "
            f"number (floating-point noise is fine, {x} is not)."
        )

    return int(rounded_x)
