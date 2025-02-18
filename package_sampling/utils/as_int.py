import numpy as np
import warnings


def as_int(x: float) -> int:
    max_int = np.iinfo(np.int32).max
    if isinstance(x, (int, np.integer)):
        return int(x)

    rounded_x = round(x)
    if x > max_int:
        raise ValueError("The input exceeds the maximum allowable integer size.")

    if not np.isclose(rounded_x, x):
        warnings.warn("The argument is not an integer", UserWarning)

    return int(rounded_x)
