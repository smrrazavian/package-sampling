from __future__ import annotations

from typing import Any, List, Union, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


def upme_pik_from_q(
    q: Union[List[List[float]], FloatArray],
    /,
) -> FloatArray:
    """
    Convert *q* matrix to *π̃* vector (R `UPMEpikfromq`).

    Parameters
    ----------
    q : 2-D array-like
        Conditional probabilities produced by :pyfunc:`upme_q_from_w`.

    Returns
    -------
    pik_tilde : 1-D ndarray
        Inclusion probabilities implied by *q*.
    """
    if not isinstance(q, np.ndarray):
        q = np.asarray(q, dtype=float)
    if q.ndim != 2:
        raise ValueError("`q` must be 2-D.")

    q = cast(FloatArray, q)
    N, n = q.shape
    if n == 0:
        return cast(FloatArray, np.zeros(N, dtype=float))

    if np.all(q == 0):
        return cast(FloatArray, np.zeros(N, dtype=float))
    if np.all(q == 1):
        return cast(FloatArray, np.ones(N, dtype=float))

    # ------------------------------------------------------------------
    # Forward–backward dynamic programme as in R but vectorised
    # ------------------------------------------------------------------
    pro = np.zeros((N, n), dtype=float)
    pro[0, n - 1] = 1.0

    for i in range(1, N):
        j = np.arange(1, n)
        pro_i_j = pro[i - 1, j]
        pro[i, j] += pro_i_j * (1.0 - q[i - 1, j])
        pro[i, j - 1] += pro_i_j * q[i - 1, j]

        pro[i, 0] += pro[i - 1, 0] * (1.0 - q[i - 1, 0])

    result = np.sum(pro * q, axis=1)
    return cast(FloatArray, result)
