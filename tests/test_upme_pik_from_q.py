import numpy as np
import pytest

from package_sampling.sampling import upme_pik_from_q


# ------------------------------------------------------------------ #
# happy-path                                                         #
# ------------------------------------------------------------------ #
def test_upmepikfromq_valid_numpy_array():
    q = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    res = upme_pik_from_q(q)

    assert res.shape == (q.shape[0],)
    assert np.all((0 <= res) & (res <= 1))


def test_upmepikfromq_accepts_list():
    q = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    res = upme_pik_from_q(q)

    assert res.shape == (len(q),)
    assert np.all((0 <= res) & (res <= 1))


# ------------------------------------------------------------------ #
# input-validation                                                   #
# ------------------------------------------------------------------ #
def test_upmepikfromq_invalid_input_not_2d():
    with pytest.raises(ValueError, match=r"`q` must be 2-D"):
        upme_pik_from_q(np.array([0.1, 0.2, 0.3]))  # 1-D array


def test_upmepikfromq_invalid_input_string():
    # NumPy raises the conversion error before our dimensionality check
    with pytest.raises(ValueError, match=r"could not convert string to float"):
        upme_pik_from_q("invalid_input")  # scalar string


def test_upmepikfromq_invalid_input_1d_list():
    with pytest.raises(ValueError, match=r"`q` must be 2-D"):
        upme_pik_from_q([0.1, 0.2, 0.3])  # 1-D list


# ------------------------------------------------------------------ #
# edge cases                                                         #
# ------------------------------------------------------------------ #
def test_upmepikfromq_edge_case_all_zeros():
    q = np.zeros((3, 3))
    res = upme_pik_from_q(q)
    assert np.all(res == 0)


def test_upmepikfromq_edge_case_all_ones():
    q = np.ones((3, 3))
    res = upme_pik_from_q(q)
    assert np.all((0 <= res) & (res <= 1))


def test_upmepikfromq_edge_case_mixed_values():
    q = np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 0.5], [0.5, 1.0, 0.0]])
    res = upme_pik_from_q(q)
    assert res.shape == (q.shape[0],)
    assert np.all((0 <= res) & (res <= 1))


def test_upmepikfromq_large_matrix():
    q = np.random.uniform(0, 1, (100, 100))
    res = upme_pik_from_q(q)
    assert res.shape == (q.shape[0],)
    assert np.all((0 <= res) & (res <= 1))

