import numpy as np
import pytest

from package_sampling.sampling import up_max_entropy


def test_upmaxentropy_valid_numpy_array():
    """Tests whether the function correctly samples from a NumPy array."""
    pik = np.array([0.1, 0.2, 0.3, 0.4])
    result = up_max_entropy(pik)

    assert result.shape == pik.shape
    assert np.all((result == 0) | (result == 1))
    assert np.isclose(np.sum(result), np.sum(pik), atol=1)


def test_upmaxentropy_accepts_list():
    """Tests whether the function correctly processes a list input."""
    pik = [0.1, 0.2, 0.3, 0.4]
    result = up_max_entropy(pik)

    assert result.shape == (len(pik),)
    assert np.all((result == 0) | (result == 1))


def test_upmaxentropy_edge_case_all_zeros():
    """Tests how the function handles a vector with all zero probabilities."""
    pik = np.zeros(5)
    result = up_max_entropy(pik)

    assert np.all(
        result == 0
    ), "The result should be all zeros for an input of all zeros."


def test_upmaxentropy_edge_case_all_ones():
    """Tests how the function handles a vector with all one probabilities."""
    pik = np.ones(5)
    result = up_max_entropy(pik)

    assert np.all(
        result == 1
    ), "The result should be all ones for an input of all ones."


def test_upmaxentropy_edge_case_sum_zero():
    """Tests how the function handles a vector whose sum is zero."""
    pik = np.array([0.0, 0.0, 0.0])
    result = up_max_entropy(pik)

    assert np.all(result == 0), "The result should be all zeros when the sum is zero."


def test_upmaxentropy_sum_preservation():
    """Tests whether the function preserves the expected sum of probabilities."""
    pik = np.array([0.25, 0.25, 0.25, 0.25])  # Sum = 1
    result = up_max_entropy(pik)

    assert np.isclose(
        np.sum(result), 1, atol=1
    ), "The sum should be approximately preserved."


def test_upmaxentropy_single_value():
    """Tests function behavior with a single inclusion probability value."""
    pik = np.array([0.7])
    result = up_max_entropy(pik)

    assert result.shape == (1,)
    assert result[0] in [0, 1]


def test_upmaxentropy_large_vector():
    """Tests function behavior on a large vector."""
    pik = np.random.uniform(0, 0.1, 100)
    result = up_max_entropy(pik)

    assert result.shape == pik.shape
    assert np.all((result == 0) | (result == 1))


def test_upmaxentropy_n_equals_one():
    """Tests behavior when n=1 (only one element should be selected)."""
    pik = np.array([0.1, 0.2, 0.3, 0.4]) / np.sum([0.1, 0.2, 0.3, 0.4])
    result = up_max_entropy(pik)

    assert np.sum(result) == 1, "Only one element should be selected."


def test_upmaxentropy_invalid_matrix_input():
    """Tests that function raises an error for non-1D inputs."""
    pik = np.array([[0.1, 0.2], [0.3, 0.4]])

    with pytest.raises(ValueError, match="pik must be a 1D vector"):
        up_max_entropy(pik)


def test_upmaxentropy_invalid_negative_values():
    """Tests that function raises an error for negative inclusion probabilities."""
    pik = np.array([0.1, -0.2, 0.3])

    with pytest.raises(
        ValueError, match="Inclusion probabilities must be between 0 and 1."
    ):
        up_max_entropy(pik)


def test_upmaxentropy_randomized_output():
    """Tests that the function provides a valid random output over multiple runs."""
    pik = np.array([0.1, 0.2, 0.3, 0.4])
    results = [up_max_entropy(pik) for _ in range(10)]

    unique_results = np.unique(results, axis=0)
    assert (
        len(unique_results) > 1
    ), "The function should produce different random samples over multiple runs."
