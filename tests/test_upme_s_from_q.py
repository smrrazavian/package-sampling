import numpy as np
import pytest

from package_sampling.sampling import upme_s_from_q


def test_upme_s_from_q_valid_input_shape():
    q = np.array([[0.1, 0.2], [0.3, 0.5], [0.6, 0.7]])
    result = upme_s_from_q(q)

    assert result.shape == (3,)
    assert np.all((result == 0) | (result == 1))
    assert result.dtype == int


def test_upme_s_from_q_input_not_2d():
    with pytest.raises(ValueError, match="Input q must be a 2D NumPy array."):
        upme_s_from_q(np.array([0.1, 0.2, 0.3]))


def test_upme_s_from_q_input_not_numpy():
    with pytest.raises(ValueError, match="Input q must be a 2D NumPy array."):
        upme_s_from_q([[0.1, 0.2], [0.3, 0.5]])


def test_upme_s_from_q_all_zero_probabilities():
    q = np.zeros((4, 3))
    result = upme_s_from_q(q)

    assert np.all(result == 0), "Expected all zero selections."


def test_upme_s_from_q_all_one_probabilities():
    q = np.ones((4, 2))
    result = upme_s_from_q(q)

    assert np.sum(result) <= 2  # Because `n = 2`, max 2 items should be selected
    assert np.all((result == 0) | (result == 1))


def test_upme_s_from_q_single_element():
    q = np.array([[0.9]])
    result = upme_s_from_q(q)

    assert result.shape == (1,)
    assert result[0] in [0, 1]


def test_upme_s_from_q_zero_columns():
    q = np.empty((5, 0))
    result = upme_s_from_q(q)

    assert np.all(result == 0), "No selections should be made when n = 0."


def test_upme_s_from_q_random_behavior_is_reasonable():
    q = np.array([[0.99, 0.99]] * 5)
    results = [upme_s_from_q(q) for _ in range(10)]

    sums = [np.sum(r) for r in results]
    assert all(s <= 2 for s in sums), "Number of selected items must not exceed n."


def test_upme_s_from_q_deterministic_with_seed():
    np.random.seed(42)
    q = np.array([[0.8, 0.9], [0.1, 0.4], [0.7, 0.6]])
    result1 = upme_s_from_q(q)

    np.random.seed(42)
    result2 = upme_s_from_q(q)

    assert np.array_equal(
        result1, result2
    ), "Output should be deterministic with same seed"
