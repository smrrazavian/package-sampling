import numpy as np
import pytest

from package_sampling import upme_q_from_w


def test_upmeqfromw_valid_input():
    """Tests whether the function correctly computes the q matrix for valid input."""
    w = np.array([0.1, 0.2, 0.3, 0.4])
    n = 2
    q = upme_q_from_w(w, n)

    assert q.shape == (len(w), n), "q matrix shape mismatch."
    assert np.all(q >= 0) and np.all(q <= 1), "Probabilities should be in [0,1] range."


def test_upmeqfromw_accepts_list():
    """Tests whether the function correctly processes a Python list as input."""
    w = [0.1, 0.2, 0.3, 0.4]
    n = 2
    q = upme_q_from_w(w, n)

    assert q.shape == (len(w), n), "q matrix shape mismatch."
    assert np.all(q >= 0) and np.all(q <= 1), "Probabilities should be in [0,1] range."


def test_upmeqfromw_invalid_n_too_large():
    """Tests whether the function raises ValueError if n is larger than the length of w."""
    w = np.array([0.1, 0.2, 0.3])
    with pytest.raises(
        ValueError, match="Sample size n cannot be larger than the length of w."
    ):
        upme_q_from_w(w, 5)


def test_upmeqfromw_edge_case_all_zeros():
    """Tests how the function handles a weight vector with all zeros."""
    w = np.zeros(4)
    n = 2
    q = upme_q_from_w(w, n)

    assert np.all(q == 0), "The q matrix should be all zeros for a zero weight vector."


def test_upmeqfromw_edge_case_all_ones():
    """Tests how the function handles a weight vector with all ones."""
    w = np.ones(4)
    n = 2
    q = upme_q_from_w(w, n)

    assert np.all(q >= 0) and np.all(
        q <= 1
    ), "All computed probabilities should be valid (0 ≤ p ≤ 1)."


def test_upmeqfromw_edge_case_large_values():
    """Tests how the function handles large values in the weight vector."""
    w = np.array([100, 200, 300, 400])
    n = 2
    q = upme_q_from_w(w, n)

    assert np.all(q >= 0) and np.all(
        q <= 1
    ), "All computed probabilities should be valid."
