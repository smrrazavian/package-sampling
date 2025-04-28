from typing import List, Union

import numpy as np


def up_poisson(pik: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Selects a sample using Poisson sampling with unequal probabilities.

    This sampling method selects each unit independently with probability pik.
    Here, the sample size is random (not fixed) with expected value of \sum_{k\in U}\pi_k.

    Args:
        pik (Union[List[float], np.ndarray]): A 1D list or NumPy array representing inclusion probabilities.
            Values should be in the range [0, 1].

    Returns:
        np.ndarray: A 1D array of the same shape as `pik`, containing only 0s and 1s.
            A `1` indicates that the corresponding element was selected in the sample.

    Raises:
        ValueError: If `pik` contains NaN values.
    """
    if not isinstance(pik, np.ndarray):
        pik = np.array(pik)

    if np.any(np.isnan(pik)):
        raise ValueError("There are missing values in the `pik` vector.")

    u = np.random.uniform(0, 1, size=len(pik))

    return np.asarray(u < pik, dtype=np.int32)
