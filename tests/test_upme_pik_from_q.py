import numpy as np
import pytest

from package_sampling import upme_pik_from_q


def test_upmepikfromq_valid_numpy_array():
    """Tests whether the function correctly computes inclusion probabilities for a NumPy array."""
    q = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    result = upme_pik_from_q(q)

    assert result.shape == (q.shape[0],)
    assert np.all(result >= 0) and np.all(result <= 1)


def test_upmepikfromq_accepts_list():
    """Tests whether the function correctly processes a list input."""
    q = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    result = upme_pik_from_q(q)

    assert result.shape == (len(q),)
    assert np.all(result >= 0) and np.all(result <= 1)


def test_upmepikfromq_invalid_input_not_2d():
    """Tests whether the function raises a ValueError for a non-2D input."""
    with pytest.raises(ValueError, match="Input q must be a 2D matrix"):
        upme_pik_from_q(np.array([0.1, 0.2, 0.3]))  # 1D array instead of 2D


def test_upmepikfromq_edge_case_all_zeros():
    """Tests how the function handles a matrix with all zero probabilities."""
    q = np.zeros((3, 3))
    result = upme_pik_from_q(q)

    assert np.all(
        result == 0
    ), "The result should be all zeros for an input of all zeros."


def test_upmepikfromq_edge_case_all_ones():
    """Tests how the function handles a matrix with all ones."""
    q = np.ones((3, 3))
    result = upme_pik_from_q(q)

    assert np.all(result >= 0) and np.all(
        result <= 1
    ), "All computed probabilities should be valid (0 ≤ p ≤ 1)."


def test_upmepikfromq_edge_case_mixed_values():
    """Tests function behavior for a probability matrix with mixed values."""
    q = np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 0.5], [0.5, 1.0, 0.0]])
    result = upme_pik_from_q(q)

    assert result.shape == (q.shape[0],)
    assert np.all(result >= 0) and np.all(
        result <= 1
    ), "All probabilities should be valid."


def test_upmepikfromq_invalid_input_string():
    """Tests whether the function raises a ValueError when input is a string."""
    with pytest.raises(ValueError, match="Input q must be a 2D matrix"):
        upme_pik_from_q("invalid_input")  # Passing a string


def test_upmepikfromq_invalid_input_1d_list():
    """Tests whether the function raises a ValueError when input is a 1D list."""
    with pytest.raises(ValueError, match="Input q must be a 2D matrix"):
        upme_pik_from_q([0.1, 0.2, 0.3])  # Passing a 1D list


def test_upmepikfromq_large_matrix():
    """Tests function behavior on a large matrix."""
    q = np.random.uniform(0, 1, (100, 100))
    result = upme_pik_from_q(q)

    assert result.shape == (q.shape[0],)
    assert np.all(result >= 0) and np.all(
        result <= 1
    ), "All probabilities should be valid."
