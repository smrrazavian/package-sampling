import numpy as np
import pytest

from package_sampling.sampling import up_poisson


# Basic tests
def test_valid_input():
    pik = [0.2, 0.5, 0.7, 0.9]
    result = up_poisson(pik)
    # Ensure the result has the same length as the input
    assert len(result) == len(pik)
    # Ensure the output is a numpy array of 0s and 1s
    assert np.all(np.isin(result, [0, 1]))


def test_empty_input():
    pik = []
    result = up_poisson(pik)
    # Empty input should return an empty array
    assert len(result) == 0


def test_na_in_pik():
    pik = [0.2, np.nan, 0.5, 0.7]
    with pytest.raises(
        ValueError, match="There are missing values in the `pik` vector."
    ):
        up_poisson(pik)


def test_all_zeros():
    pik = [0.0, 0.0, 0.0, 0.0]
    result = up_poisson(pik)
    # With all zeros, no units should be selected
    assert np.sum(result) == 0


def test_all_ones():
    pik = [1.0, 1.0, 1.0, 1.0]
    result = up_poisson(pik)
    # With all ones, all units should be selected
    assert np.sum(result) == len(pik)


def test_extreme_values():
    pik = [0.0, 1.0, 0.5]
    result = up_poisson(pik)
    # The first unit should never be selected
    assert result[0] == 0
    # The second unit should always be selected
    assert result[1] == 1
    # The third unit is probabilistic, so we don't test it


# Statistical property tests
def test_poisson_expected_sample_size():
    """Test that Poisson sampling has expected sample size close to sum of probabilities."""
    pik = np.array([0.2, 0.3, 0.5])
    expected_size = np.sum(pik)

    num_simulations = 1000
    sample_sizes = []

    for i in range(num_simulations):
        np.random.seed(i)
        result = up_poisson(pik)
        sample_sizes.append(np.sum(result))

    mean_sample_size = np.mean(sample_sizes)
    # Check if the mean sample size is close to expected (within 10%)
    assert np.isclose(
        mean_sample_size, expected_size, rtol=0.1
    ), f"Expected mean sample size {expected_size}, got {mean_sample_size}"


def test_poisson_inclusion_probabilities():
    """Test if actual inclusion probabilities match theoretical ones."""
    pik = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    num_simulations = 5000

    results = np.zeros((num_simulations, len(pik)))
    for i in range(num_simulations):
        np.random.seed(i)
        results[i] = up_poisson(pik)

    actual_probs = results.mean(axis=0)
    std_error = np.sqrt(pik * (1 - pik) / num_simulations)

    # Check if actual probabilities are within 3 standard errors of theoretical
    within_bounds = np.abs(actual_probs - pik) <= 3 * std_error

    print(f"Expected probabilities: {pik}")
    print(f"Actual probabilities: {actual_probs}")
    print(f"Std Error: {std_error}")

    assert np.all(
        within_bounds
    ), f"Inclusion probabilities don't match expected: {list(zip(pik, actual_probs))}"


def test_poisson_independence():
    """Test that selections are independent in Poisson sampling."""
    pik = np.array([0.3, 0.4, 0.5])
    num_simulations = 5000

    results = np.zeros((num_simulations, len(pik)))
    for i in range(num_simulations):
        np.random.seed(i)
        results[i] = up_poisson(pik)

    # Calculate joint selection probabilities
    joint_probs = np.zeros((len(pik), len(pik)))
    for i in range(len(pik)):
        for j in range(i + 1, len(pik)):
            joint_probs[i, j] = np.mean(
                np.logical_and(results[:, i] == 1, results[:, j] == 1)
            )
            joint_probs[j, i] = joint_probs[i, j]

    # Calculate marginal selection probabilities
    marginal_probs = results.mean(axis=0)

    # For independence, joint probability should approximately equal product of marginals
    for i in range(len(pik)):
        for j in range(i + 1, len(pik)):
            expected_joint = marginal_probs[i] * marginal_probs[j]
            actual_joint = joint_probs[i, j]

            # Calculate standard error for the joint probability
            p_joint = expected_joint
            se_joint = np.sqrt(p_joint * (1 - p_joint) / num_simulations)

            # Check if within 3 standard errors
            assert np.abs(actual_joint - expected_joint) <= 3 * se_joint, (
                f"Units {i} and {j} selections are not independent. "
                f"Expected joint prob: {expected_joint}, actual: {actual_joint}"
            )


def test_poisson_variance():
    """Test the variance of the sample size in Poisson sampling."""
    pik = np.array([0.2, 0.3, 0.5])
    expected_variance = np.sum(
        pik * (1 - pik)
    )  # Theoretical variance for Poisson sampling

    num_simulations = 10000
    sample_sizes = []

    for i in range(num_simulations):
        np.random.seed(i)
        result = up_poisson(pik)
        sample_sizes.append(np.sum(result))

    observed_variance = np.var(sample_sizes)

    # Check if the observed variance is close to expected (within 10%)
    assert np.isclose(
        observed_variance, expected_variance, rtol=0.1
    ), f"Expected variance {expected_variance}, got {observed_variance}"


@pytest.mark.parametrize(
    "pik",
    [
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([0.01, 0.99, 0.5, 0.5]),
        np.array([0.33, 0.33, 0.33]),
    ],
)
def test_poisson_with_various_distributions(pik):
    """Test Poisson sampling with different probability distributions."""
    expected_size = np.sum(pik)
    num_simulations = 1000

    sample_sizes = []
    for i in range(num_simulations):
        np.random.seed(i)
        result = up_poisson(pik)
        sample_sizes.append(np.sum(result))

    mean_sample_size = np.mean(sample_sizes)

    # Check if the mean sample size is close to expected (within 10%)
    assert np.isclose(
        mean_sample_size, expected_size, rtol=0.1
    ), f"Expected mean sample size {expected_size}, got {mean_sample_size}"


@pytest.mark.slow
def test_intensive_poisson_properties():
    """More intensive test of Poisson sampling properties."""
    pik = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    num_simulations = 10000

    # Run many simulations
    results = np.zeros((num_simulations, len(pik)))
    for i in range(num_simulations):
        np.random.seed(i)
        result = up_poisson(pik)
        results[i] = result

    # Calculate actual inclusion probabilities
    actual_probs = results.mean(axis=0)

    # Calculate sample size statistics
    sample_sizes = np.sum(results, axis=1)
    mean_sample_size = np.mean(sample_sizes)
    var_sample_size = np.var(sample_sizes)

    # Calculate joint probabilities
    joint_probs = np.zeros((len(pik), len(pik)))
    for i in range(len(pik)):
        for j in range(len(pik)):
            if i == j:
                joint_probs[i, j] = actual_probs[i]  # Probability of selecting unit i
            else:
                joint_probs[i, j] = np.mean(
                    np.logical_and(results[:, i] == 1, results[:, j] == 1)
                )

    # Expected values for Poisson sampling
    expected_size = np.sum(pik)
    expected_var = np.sum(pik * (1 - pik))

    print(f"Expected probabilities: {pik}")
    print(f"Actual probabilities: {actual_probs}")
    print(f"Expected mean sample size: {expected_size}")
    print(f"Actual mean sample size: {mean_sample_size}")
    print(f"Expected variance: {expected_var}")
    print(f"Actual variance: {var_sample_size}")
    print(f"Joint probability matrix:\n{joint_probs}")

    # Check mean sample size
    assert np.isclose(
        mean_sample_size, expected_size, rtol=0.05
    ), f"Expected mean sample size {expected_size}, got {mean_sample_size}"

    # Check variance
    assert np.isclose(
        var_sample_size, expected_var, rtol=0.1
    ), f"Expected variance {expected_var}, got {var_sample_size}"

    # Check inclusion probabilities
    assert np.all(
        np.isclose(actual_probs, pik, rtol=0.1)
    ), "Inclusion probabilities don't match expected"

    # Check independence
    for i in range(len(pik)):
        for j in range(i + 1, len(pik)):
            expected_joint = pik[i] * pik[j]
            actual_joint = joint_probs[i, j]
            assert np.isclose(
                actual_joint, expected_joint, rtol=0.1
            ), f"Joint probability for units {i},{j} doesn't match expected"
