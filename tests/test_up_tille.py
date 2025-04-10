import numpy as np
import pytest

from package_sampling.sampling import up_tille


def test_valid_input():
    pik = [0.2, 0.5, 0.7, 0.9]
    result = up_tille(pik)
    # Ensure the result has the same length as the input
    assert len(result) == len(pik)
    # Ensure the output is a numpy array of 0s and 1s
    assert np.all(np.isin(result, [0, 1]))


def test_edge_case_all_out_of_range():
    pik = [0.0, 1.0]
    with pytest.raises(ValueError, match="All elements in `pik` are outside the range"):
        up_tille(pik)


def test_empty_input():
    pik = []
    result = up_tille(pik)
    # Empty input should return an empty array
    assert len(result) == 0


def test_na_in_pik():
    pik = [0.2, np.nan, 0.5, 0.7]
    with pytest.raises(
        ValueError, match="There are missing values in the `pik` vector."
    ):
        up_tille(pik)


def test_all_same_probability():
    pik = [0.5, 0.5, 0.5, 0.5]
    result = up_tille(pik)
    # The result should still be an array of 0s and 1s
    assert len(result) == len(pik)
    assert np.all(np.isin(result, [0, 1]))


def test_high_eps_value():
    pik = [0.9, 0.5, 0.7, 0.3]
    eps = 1e-2
    result = up_tille(pik, eps=eps)
    assert len(result) == len(pik)
    assert np.all(np.isin(result, [0, 1]))


def test_low_eps_value():
    pik = [0.2, 0.5, 0.7, 0.9]
    eps = 1e-8  # smaller eps to test the effect of higher precision
    result = up_tille(pik, eps=eps)
    assert len(result) == len(pik)
    assert np.all(np.isin(result, [0, 1]))


def test_probability_edge_cases():
    pik = [1e-7, 0.9999999]
    with pytest.raises(ValueError, match="All elements in `pik` are outside the range"):
        up_tille(pik)
