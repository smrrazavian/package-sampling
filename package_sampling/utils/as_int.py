import warnings

import numpy as np


def as_int(x: float) -> int:
    """
    Converts a floating-point number to an integer safely.

    Args:
        x (float): The number to convert.

    Returns:
        int: The integer representation of `x`.

    Raises:
        ValueError: If `x` exceeds the maximum allowable integer size.
        UserWarning: If `x` is not close to an integer.
    """
    int_min, int_max = np.iinfo(np.int32).min, np.iinfo(np.int32).max

    if isinstance(x, (int, np.integer)):
        return int(x)

    if x < int_min or x > int_max:
        raise ValueError(
            f"The input {x} exceeds the allowable integer range [{int_min}, {int_max}]."
        )

    rounded_x = round(x)

    if not np.isclose(rounded_x, x, atol=1e-6):
        warnings.warn(
            f"The argument {x} is not exactly an integer. Using rounded value {rounded_x}.",
            UserWarning,
        )

    return int(rounded_x)
