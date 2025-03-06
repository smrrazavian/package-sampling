import pytest
import numpy as np
from package_sampling import UPTille


@pytest.fixture
def uptille():
    return UPTille()


def test_valid_input(uptille):
    pik = [0.2, 0.5, 0.7, 0.9]
    result = uptille.sample(pik)
    # Ensure the result has the same length as the input
    assert len(result) == len(pik)
    # Ensure the output is a numpy array of 0s and 1s
    assert np.all(np.isin(result, [0, 1]))


def test_edge_case_all_out_of_range(uptille):
    pik = [0.0, 1.0]
    with pytest.raises(ValueError, match="All elements in `pik` are outside the range"):
        uptille.sample(pik)


def test_empty_input(uptille):
    pik = []
    result = uptille.sample(pik)
    # Empty input should return an empty array
    assert len(result) == 0


def test_na_in_pik(uptille):
    pik = [0.2, np.nan, 0.5, 0.7]
    with pytest.raises(
        ValueError, match="There are missing values in the `pik` vector."
    ):
        uptille.sample(pik)


def test_all_same_probability(uptille):
    pik = [0.5, 0.5, 0.5, 0.5]
    result = uptille.sample(pik)
    # The result should still be an array of 0s and 1s
    assert len(result) == len(pik)
    assert np.all(np.isin(result, [0, 1]))


def test_high_eps_value(uptille):
    pik = [0.9, 0.5, 0.7, 0.3]
    eps = 1e-2
    result = uptille.sample(pik, eps=eps)
    assert len(result) == len(pik)
    assert np.all(np.isin(result, [0, 1]))


def test_low_eps_value(uptille):
    pik = [0.2, 0.5, 0.7, 0.9]
    eps = 1e-8  # smaller eps to test the effect of higher precision
    result = uptille.sample(pik, eps=eps)
    assert len(result) == len(pik)
    assert np.all(np.isin(result, [0, 1]))


def test_probability_edge_cases(uptille):
    pik = [1e-7, 0.9999999]
    with pytest.raises(ValueError, match="All elements in `pik` are outside the range"):
        uptille.sample(pik)
