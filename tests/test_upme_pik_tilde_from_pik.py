import numpy as np

from package_sampling import upme_pik_tilde_from_pik


def test_upmepiktildefrompik_valid_numpy_array():
    """Tests whether the function correctly computes adjusted inclusion probabilities for a NumPy array."""
    pik = np.array([0.1, 0.2, 0.3, 0.4])
    result = upme_pik_tilde_from_pik(pik)
    assert result.shape == pik.shape
    assert np.all(result >= 0) and np.all(result <= 1)


def test_upmepiktildefrompik_accepts_list():
    """Tests whether the function correctly processes a list input."""
    pik = [0.1, 0.2, 0.3, 0.4]
    result = upme_pik_tilde_from_pik(pik)
    assert result.shape == (len(pik),)
    assert np.all(result >= 0) and np.all(result <= 1)


def test_upmepiktildefrompik_convergence():
    """Tests whether the function converges properly."""
    pik = np.array([0.2, 0.3, 0.5])
    result = upme_pik_tilde_from_pik(pik)
    assert np.isclose(np.sum(result), np.sum(pik), atol=1e-5)


def test_upmepiktildefrompik_custom_eps():
    """Tests function behavior with a custom epsilon value."""
    pik = np.array([0.2, 0.3, 0.5])
    result = upme_pik_tilde_from_pik(pik, eps=1e-8)
    assert np.all(result >= 0) and np.all(result <= 1)


def test_upmepiktildefrompik_edge_case_all_zeros():
    """Tests how the function handles a vector with all zero probabilities."""
    pik = np.zeros(5)
    result = upme_pik_tilde_from_pik(pik)
    assert np.all(
        result == 0
    ), "The result should be all zeros for an input of all zeros."


def test_upmepiktildefrompik_edge_case_sum_equals_zero():
    """Tests how the function handles a vector whose sum is zero."""
    pik = np.array([0.0, 0.0, 0.0, 0.0])
    result = upme_pik_tilde_from_pik(pik)
    assert np.all(result == 0), "The result should be all zeros when the sum is zero."


def test_upmepiktildefrompik_edge_case_ones():
    """Tests how the function handles boundary probabilities of 1."""
    pik = np.array([0.2, 0.3, 1.0, 0.5])
    result = upme_pik_tilde_from_pik(pik)
    assert np.all(result >= 0) and np.all(result <= 1)


def test_upmepiktildefrompik_empty_input():
    """Tests how the function handles an empty input."""
    pik = np.array([])
    result = upme_pik_tilde_from_pik(pik)
    assert result.shape == (0,)


def test_upmepiktildefrompik_large_vector():
    """Tests function behavior on a large vector."""
    pik = np.random.uniform(0, 0.01, 100)
    result = upme_pik_tilde_from_pik(pik)
    assert result.shape == pik.shape
    assert np.all(result >= 0) and np.all(result <= 1)


def test_upmepiktildefrompik_values_outside_range():
    """Tests how the function handles values outside the [0,1] range."""
    pik = np.array([-0.1, 0.5, 1.2])
    # Check that it doesn't crash, but we don't validate specific output since the behavior
    # for out-of-range values might need to be defined by your implementation
    result = upme_pik_tilde_from_pik(pik)
    assert result.shape == pik.shape


def test_upmepiktildefrompik_single_value():
    """Tests function behavior with a single value."""
    pik = np.array([0.5])
    result = upme_pik_tilde_from_pik(pik)
    assert result.shape == (1,)
    assert 0 <= result[0] <= 1


def test_upmepiktildefrompik_sum_equals_integer():
    """Tests whether the function preserves the sum when it's an integer."""
    pik = np.array([0.25, 0.25, 0.25, 0.25])  # Sum = 1
    result = upme_pik_tilde_from_pik(pik)
    assert np.isclose(np.sum(result), 1.0, atol=1e-5)
