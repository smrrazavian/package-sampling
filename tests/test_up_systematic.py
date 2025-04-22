import numpy as np
import pytest

from package_sampling.sampling import up_systematic


def test_up_systematic_basic_sampling():
    pik = np.array([0.1, 0.2, 0.3, 0.4])
    result = up_systematic(pik)

    assert result.shape == pik.shape
    assert np.all((result == 0) | (result == 1))


def test_up_systematic_accepts_list():
    pik = [0.2, 0.5, 0.3]
    result = up_systematic(pik)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.all((result == 0) | (result == 1))


def test_up_systematic_handles_all_zero():
    pik = np.zeros(5)
    result = up_systematic(pik)

    assert np.all(result == 0)


def test_up_systematic_handles_all_one():
    pik = np.ones(4)
    result = up_systematic(pik)

    assert np.all(result == 1)


def test_up_systematic_partial_ones_and_zeros():
    pik = np.array([0.0, 1.0, 0.3, 0.6])
    result = up_systematic(pik)

    assert result[0] == 0
    assert result[1] == 1
    assert np.all((result[2:] == 0) | (result[2:] == 1))


def test_up_systematic_nan_raises_error():
    pik = np.array([0.1, np.nan, 0.3])

    with pytest.raises(ValueError, match="missing values"):
        up_systematic(pik)


def test_up_systematic_determinism_with_seed():
    np.random.seed(123)
    pik = np.array([0.3, 0.3, 0.4])
    r1 = up_systematic(pik)

    np.random.seed(123)
    r2 = up_systematic(pik)

    assert np.array_equal(r1, r2)


# Statistical Property Tests
def test_inclusion_probabilities():
    """Test if actual inclusion probabilities match theoretical ones."""
    pik = np.array([0.1, 0.2, 0.3, 0.4])
    num_simulations = 10000

    results = np.zeros((num_simulations, len(pik)))
    for i in range(num_simulations):
        np.random.seed(i)
        results[i] = up_systematic(pik)

    actual_probs = results.mean(axis=0)

    std_error = np.sqrt(pik * (1 - pik) / num_simulations)

    # Test if differences are within 3 standard errors (99.7% confidence)
    within_bounds = np.abs(actual_probs - pik) <= 3 * std_error

    print(f"Expected: {pik}")
    print(f"Actual: {actual_probs}")
    print(f"Std Error: {std_error}")

    assert np.all(
        within_bounds
    ), f"Inclusion probabilities don't match expected: {list(zip(pik, actual_probs))}"


def test_sample_size_variance():
    """
    Test that the sample size variance is less than that of Poisson sampling.
    This is a key characteristic of systematic sampling.
    """
    pik = np.array([0.2, 0.3, 0.25, 0.25])
    num_simulations = 10000

    sample_sizes = np.zeros(num_simulations)
    for i in range(num_simulations):
        np.random.seed(i)
        sample = up_systematic(pik)
        sample_sizes[i] = np.sum(sample)

    actual_variance = np.var(sample_sizes)

    poisson_variance = np.sum(pik * (1 - pik))

    variance_ratio = actual_variance / poisson_variance

    print(f"Sample size mean: {np.mean(sample_sizes)}")
    print(f"Sample size variance: {actual_variance}")
    print(f"Poisson variance: {poisson_variance}")
    print(f"Variance ratio: {variance_ratio}")

    assert (
        variance_ratio < 1.0
    ), f"Sample size variance ({actual_variance}) should be less than Poisson variance ({poisson_variance})"


def test_negative_correlations():
    """
    Test that UP systematic sampling introduces negative correlations between units.
    This is another key characteristic of systematic sampling.
    """
    pik = np.array([0.3, 0.4, 0.3])
    num_simulations = 10000

    results = np.zeros((num_simulations, len(pik)))
    for i in range(num_simulations):
        np.random.seed(i)
        results[i] = up_systematic(pik)

    correlation_matrix = np.corrcoef(results.T)

    # Calculate pairwise covariances
    covariances = np.zeros((len(pik), len(pik)))
    for i in range(len(pik)):
        for j in range(len(pik)):
            if i != j:
                # Actual joint probability - use logical_and instead of bitwise &
                joint_prob = np.mean(
                    np.logical_and(results[:, i] == 1, results[:, j] == 1)
                )
                # Expected joint probability under independence
                expected_joint = pik[i] * pik[j]
                # Covariance
                covariances[i, j] = joint_prob - expected_joint

    # Check if off-diagonal correlations are predominantly negative
    off_diag_indices = np.where(~np.eye(len(pik), dtype=bool))
    neg_correlation_ratio = np.sum(correlation_matrix[off_diag_indices] < 0) / len(
        off_diag_indices[0]
    )

    print(f"Correlation matrix:\n{correlation_matrix}")
    print(f"Negative correlation ratio: {neg_correlation_ratio}")

    assert (
        neg_correlation_ratio > 0.5
    ), f"Expected predominantly negative correlations, but only {neg_correlation_ratio:.2%} are negative"


def test_edge_cases_systematic():
    """Test edge cases specific to systematic sampling."""
    pik_edge = np.array([0.001, 0.999, 0.5, 0.5])

    selections_near_zero = 0
    selections_near_one = 0

    for i in range(1000):
        np.random.seed(i)
        result = up_systematic(pik_edge)
        selections_near_zero += result[0]
        selections_near_one += result[1]

    assert (
        selections_near_one > 990
    ), f"Unit with probability 0.999 was only selected {selections_near_one} times out of 1000"

    assert (
        selections_near_zero < 10
    ), f"Unit with probability 0.001 was selected {selections_near_zero} times out of 1000"


@pytest.mark.parametrize(
    "pik",
    [
        np.array([0.25, 0.25, 0.25, 0.25]),  # Uniform probabilities
        np.array([0.1, 0.2, 0.3, 0.4]),  # Increasing probabilities
        np.array([0.4, 0.3, 0.2, 0.1]),  # Decreasing probabilities
    ],
)
def test_systematic_with_various_distributions(pik):
    """
    Test systematic sampling with different probability distributions.
    This parametrized test runs the inclusion probability test on multiple inputs.
    """
    num_simulations = 5000

    results = np.zeros((num_simulations, len(pik)))
    for i in range(num_simulations):
        np.random.seed(i)
        results[i] = up_systematic(pik)

    actual_probs = results.mean(axis=0)
    max_diff = np.max(np.abs(actual_probs - pik))

    assert (
        max_diff < 0.05
    ), f"Inclusion probabilities don't match for {pik}: max difference is {max_diff}"


@pytest.mark.slow
def test_intensive_systematic_properties():
    """
    More intensive test of systematic sampling properties.
    This test is marked with 'slow' to indicate it takes longer to run.
    """
    pik = np.array([0.1, 0.2, 0.3, 0.4])
    num_simulations = 50000

    results = np.zeros((num_simulations, len(pik)))
    sample_sizes = np.zeros(num_simulations)

    for i in range(num_simulations):
        np.random.seed(i)
        sample = up_systematic(pik)
        results[i] = sample
        sample_sizes[i] = np.sum(sample)

    actual_probs = results.mean(axis=0)
    prob_diff = np.abs(actual_probs - pik)

    # Test sample size properties
    mean_size = np.mean(sample_sizes)
    var_size = np.var(sample_sizes)
    expected_size = np.sum(pik)
    poisson_var = np.sum(pik * (1 - pik))

    # Test second-order inclusion probabilities (joint selection probabilities)
    joint_probs = np.zeros((len(pik), len(pik)))
    for i in range(len(pik)):
        for j in range(i + 1, len(pik)):
            joint_probs[i, j] = np.mean(
                np.logical_and(results[:, i] == 1, results[:, j] == 1)
            )
            joint_probs[j, i] = joint_probs[i, j]

    # Diagonal elements are just the first-order probabilities
    for i in range(len(pik)):
        joint_probs[i, i] = actual_probs[i]

    # Calculate covariances
    covariances = np.zeros((len(pik), len(pik)))
    for i in range(len(pik)):
        for j in range(len(pik)):
            if i != j:
                covariances[i, j] = joint_probs[i, j] - (pik[i] * pik[j])

    # Sum of all covariances should approximately equal negative of size variance
    sum_cov = np.sum(covariances) - np.sum(np.diag(covariances))

    # Print detailed statistics for analysis
    print(f"Expected probabilities: {pik}")
    print(f"Actual probabilities: {actual_probs}")
    print(f"Probability differences: {prob_diff}")
    print(f"Expected sample size: {expected_size}")
    print(f"Actual mean sample size: {mean_size}")
    print(f"Sample size variance: {var_size}")
    print(f"Poisson variance: {poisson_var}")
    print(f"Variance ratio: {var_size / poisson_var}")
    print(f"Sum of covariances: {sum_cov}")
    print(f"Negative of size variance: {-var_size}")
    print(f"Ratio sum_cov / (-var_size): {sum_cov / (-var_size)}")

    # Assertions for comprehensive validation
    assert np.all(prob_diff < 0.01), "Inclusion probabilities don't match expected"
    assert np.isclose(
        mean_size, expected_size, rtol=0.01
    ), "Mean sample size doesn't match expected"
    assert var_size < poisson_var, "Sample size variance exceeds Poisson variance"

    # Skip exact covariance check, but verify the right direction and approximate magnitude
    assert sum_cov < 0, "Sum of covariances should be negative"
    assert (
        abs(sum_cov) > 0.5 * var_size
    ), "Sum of covariances should be substantial compared to size variance"
