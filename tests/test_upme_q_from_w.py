from __future__ import annotations

import numpy as np
import pytest

from package_sampling.sampling import upme_q_from_w


# ---------------------------------------------------------------------- #
# happy-path                                                             #
# ---------------------------------------------------------------------- #
def test_upmeqfromw_valid_input():
    w = np.array([0.1, 0.2, 0.3, 0.4])
    q = upme_q_from_w(w, 2)

    assert q.shape == (len(w), 2)
    assert np.all((0 <= q) & (q <= 1))


def test_upmeqfromw_accepts_python_list():
    w = [0.1, 0.2, 0.3, 0.4]
    q = upme_q_from_w(w, 2)
    assert q.shape == (4, 2)
    assert np.all((0 <= q) & (q <= 1))


# ---------------------------------------------------------------------- #
# input validation                                                       #
# ---------------------------------------------------------------------- #
def test_upmeqfromw_invalid_n_too_large():
    w = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match=r"`n` out of range"):
        upme_q_from_w(w, 5)


def test_upmeqfromw_input_not_1d():
    w = np.array([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="`w` must be 1-D"):
        upme_q_from_w(w, 1)


# ---------------------------------------------------------------------- #
# edge cases                                                             #
# ---------------------------------------------------------------------- #
def test_upmeqfromw_edge_case_all_zeros():
    q = upme_q_from_w(np.zeros(4), 2)
    assert np.all(q == 0)


def test_upmeqfromw_edge_case_all_ones():
    q = upme_q_from_w(np.ones(4), 2)
    assert np.all((0 <= q) & (q <= 1))


def test_upmeqfromw_edge_case_large_values():
    # integers are fine – the function converts to float internally
    w = np.array([100, 200, 300, 400], dtype=float)
    q = upme_q_from_w(w, 2)
    assert np.all((0 <= q) & (q <= 1))
