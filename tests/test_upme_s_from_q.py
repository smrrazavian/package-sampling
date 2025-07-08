from __future__ import annotations

import numpy as np
import pytest

from package_sampling.sampling import upme_s_from_q


# ---------------------------------------------------------------------- #
# happy-path                                                             #
# ---------------------------------------------------------------------- #
def test_upme_s_from_q_valid_input_shape():
    q = np.array([[0.1, 0.2], [0.3, 0.5], [0.6, 0.7]])
    result = upme_s_from_q(q)

    assert result.shape == (3,)
    assert np.all((result == 0) | (result == 1))
    # the routine promises int8 – but accept any integer subtype
    assert np.issubdtype(result.dtype, np.integer)


# ---------------------------------------------------------------------- #
# input validation                                                       #
# ---------------------------------------------------------------------- #
def test_upme_s_from_q_input_not_2d_array():
    with pytest.raises(ValueError, match=r"`q` must be a 2-D NumPy array\."):
        upme_s_from_q(np.array([0.1, 0.2, 0.3]))


def test_upme_s_from_q_input_not_numpy():
    with pytest.raises(ValueError, match=r"`q` must be a 2-D NumPy array\."):
        upme_s_from_q([[0.1, 0.2], [0.3, 0.5]])  # plain Python list


# ---------------------------------------------------------------------- #
# edge cases                                                             #
# ---------------------------------------------------------------------- #
def test_upme_s_from_q_all_zero_probabilities():
    q = np.zeros((4, 3))
    result = upme_s_from_q(q)
    assert np.all(result == 0)


def test_upme_s_from_q_all_one_probabilities():
    q = np.ones((4, 2))
    result = upme_s_from_q(q)
    assert result.sum() <= 2  # n = 2, at most two selections
    assert np.all((result == 0) | (result == 1))


def test_upme_s_from_q_single_element():
    q = np.array([[0.9]])
    result = upme_s_from_q(q)
    assert result.shape == (1,)
    assert result[0] in (0, 1)


def test_upme_s_from_q_zero_columns():
    q = np.empty((5, 0))
    assert np.all(upme_s_from_q(q) == 0)


# ---------------------------------------------------------------------- #
# stochastic behaviour                                                   #
# ---------------------------------------------------------------------- #
def test_upme_s_from_q_random_behavior_is_reasonable():
    q = np.array([[0.99, 0.99]] * 5)  # n = 2
    draws = [upme_s_from_q(q) for _ in range(20)]
    assert all(d.sum() <= 2 for d in draws)


def test_upme_s_from_q_deterministic_with_rng():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    q = np.array([[0.8, 0.9], [0.1, 0.4], [0.7, 0.6]])

    res1 = upme_s_from_q(q, rng=rng1)
    res2 = upme_s_from_q(q, rng=rng2)
    assert np.array_equal(res1, res2)
