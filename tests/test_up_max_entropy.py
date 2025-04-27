import numpy as np
import pytest

from package_sampling.sampling import up_max_entropy
from package_sampling.utils import inclusion_probabilities


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


def test_maxentropy_inclusion_probabilities():
    """
    Test if actual inclusion probabilities match theoretical ones
    over many simulations.
    """
    # Define inclusion probabilities
    pik = np.array([0.2, 0.3, 0.5])

    # Run many simulations
    num_simulations = 10000
    results = np.zeros((num_simulations, len(pik)))

    for i in range(num_simulations):
        np.random.seed(i)  # For reproducibility
        results[i] = up_max_entropy(pik)

    # Calculate actual inclusion probabilities
    actual_probs = np.mean(results, axis=0)

    # Calculate standard errors
    std_error = np.sqrt(pik * (1 - pik) / num_simulations)

    # Check if actual probabilities are within 3 standard errors of expected
    within_bounds = np.abs(actual_probs - pik) <= 3 * std_error

    # Debug prints
    print(f"Expected inclusion probabilities: {pik}")
    print(f"Actual inclusion probabilities:   {actual_probs}")
    print(f"Standard errors:                  {std_error}")
    print(f"Differences:                      {np.abs(actual_probs - pik)}")
    print(f"Within bounds:                    {within_bounds}")

    assert np.all(
        within_bounds
    ), f"Inclusion probabilities don't match expected: {list(zip(pik, actual_probs))}"


def test_maxentropy_fixed_sample_size():
    """
    Test that maximum entropy sampling produces samples with the expected size.
    """
    # Test with different probability vectors
    test_cases = [
        np.array([0.2, 0.3, 0.5]),  # Sum = 1
        np.array([0.3, 0.3, 0.3, 0.3]),  # Sum = 1.2
        np.array([0.1, 0.2, 0.3]),  # Sum = 0.6
    ]

    for pik in test_cases:
        expected_size = int(np.round(np.sum(pik)))

        # Run multiple simulations
        sample_sizes = []
        for i in range(100):
            np.random.seed(i)
            result = up_max_entropy(pik)
            sample_sizes.append(np.sum(result))

        # Debug prints for fixed sample size test
        unique_sizes = np.unique(sample_sizes)
        size_counts = {size: sample_sizes.count(size) for size in unique_sizes}

        print(f"\nProbability vector: {pik}")
        print(f"Expected sample size: {expected_size}")
        print(f"Observed sample sizes: {size_counts}")

        # Check if all sample sizes are as expected
        assert np.all(
            np.array(sample_sizes) == expected_size
        ), f"Expected fixed sample size {expected_size}, but got sizes: {np.unique(sample_sizes)}"


@pytest.mark.slow
def test_maxentropy_joint_probabilities():
    """
    Test second-order inclusion probabilities (joint selection probabilities).
    """
    pik = np.array([0.2, 0.4, 0.4])
    num_simulations = 10000

    results = np.zeros((num_simulations, len(pik)))
    for i in range(num_simulations):
        np.random.seed(i)
        results[i] = up_max_entropy(pik)

    # First-order probabilities (marginal)
    first_order = np.mean(results, axis=0)

    # Calculate joint selection probabilities
    joint_probs = np.zeros((len(pik), len(pik)))
    for i in range(len(pik)):
        for j in range(i + 1, len(pik)):
            joint_probs[i, j] = np.mean(
                np.logical_and(results[:, i] == 1, results[:, j] == 1)
            )
            joint_probs[j, i] = joint_probs[i, j]  # Symmetric matrix

    # Set diagonal to first-order probabilities
    for i in range(len(pik)):
        joint_probs[i, i] = first_order[i]

    # Debug prints
    print(f"Input probabilities: {pik}")
    print(f"First-order inclusion probabilities: {first_order}")
    print(f"Joint probabilities matrix:")
    print(joint_probs)
    print(f"Products of marginal probabilities:")
    for i in range(len(pik)):
        for j in range(i + 1, len(pik)):
            print(
                f"P({i},{j}): Joint={joint_probs[i,j]:.4f}, Product={first_order[i]*first_order[j]:.4f}, Diff={joint_probs[i,j]-first_order[i]*first_order[j]:.4f}"
            )

    # Check maximum entropy property: joint probabilities should be close to
    # product of marginals when possible (for sample size > 1)
    expected_size = int(np.round(np.sum(pik)))

    if expected_size > 1:
        for i in range(len(pik)):
            for j in range(i + 1, len(pik)):
                product_of_probs = first_order[i] * first_order[j]

                # Maximum entropy tries to make joint probs close to product of marginals
                # when consistent with fixed sample size
                # We use a relatively large tolerance here due to the finite sample
                assert (
                    np.abs(joint_probs[i, j] - product_of_probs) < 0.05
                ), f"Joint probabilities don't reflect maximum entropy for units {i} and {j}"


@pytest.mark.slow
def test_maxentropy_with_certain_selections():
    """
    Test behavior when some units have inclusion probability of 1.
    These units should always be selected.
    """
    pik = np.array([1.0, 0.2, 0.3, 0.5])

    # Run multiple simulations
    results = np.zeros((1000, len(pik)))
    for i in range(1000):
        np.random.seed(i)
        results[i] = up_max_entropy(pik)

    # Units with pik=1 should always be selected
    always_selected = np.all(results[:, 0] == 1)
    assert (
        always_selected
    ), "Units with inclusion probability 1 should always be selected"

    # Check other units' selection rates
    other_probs = np.mean(results[:, 1:], axis=0)
    expected_probs = np.array([0.2, 0.3, 0.5])

    # Calculate standard errors
    std_error = np.sqrt(expected_probs * (1 - expected_probs) / 1000)

    # Check if actual probabilities are within 3 standard errors of expected
    within_bounds = np.abs(other_probs - expected_probs) <= 3 * std_error

    # Debug prints
    print(f"Units with certainty (pik=1): Always selected = {always_selected}")
    print(f"Expected other probabilities: {expected_probs}")
    print(f"Actual other probabilities:   {other_probs}")
    print(f"Standard errors:              {std_error}")
    print(f"Differences:                  {np.abs(other_probs - expected_probs)}")
    print(f"Within bounds:                {within_bounds}")

    assert np.all(
        within_bounds
    ), f"Inclusion probabilities don't match expected: {list(zip(expected_probs, other_probs))}"


@pytest.mark.slow
def test_maxentropy_entropy_maximization():
    """
    Test that the entropy of the sampling design is maximized.

    For fixed first-order inclusion probabilities, the maximum entropy design
    minimizes the variance of the Horvitz-Thompson estimator.

    We can compare the variance of our estimator with a simpler design
    (e.g., Poisson sampling) that has the same first-order inclusion probabilities.
    """
    # Define a population with values
    population = np.array([10, 20, 30, 40])
    N = len(population)

    # Define inclusion probabilities proportional to values
    pik = population / np.sum(population) * 2  # Sample size approximately 2

    # Simulate many samples using maximum entropy
    num_simulations = 10000
    maxent_estimates = []

    for i in range(num_simulations):
        np.random.seed(i)
        s = up_max_entropy(pik)

        # Skip samples where no units are selected
        if np.sum(s) == 0:
            continue

        # Calculate Horvitz-Thompson estimate
        ht_estimate = np.sum(population[s == 1] / pik[s == 1])
        maxent_estimates.append(ht_estimate)

    # Calculate variance of the estimator using maximum entropy
    maxent_variance = np.var(maxent_estimates)

    # Simulate independent Poisson sampling (not fixed size)
    poisson_estimates = []
    for i in range(num_simulations):
        np.random.seed(i + num_simulations)  # Different seeds
        s_poisson = np.random.binomial(1, pik)

        # Skip samples where no units are selected
        if np.sum(s_poisson) == 0:
            continue

        ht_estimate = np.sum(population[s_poisson == 1] / pik[s_poisson == 1])
        poisson_estimates.append(ht_estimate)

    # Calculate variance of the Poisson estimator
    poisson_variance = np.var(poisson_estimates)

    # Debug prints for variance comparison
    print(f"Population values:             {population}")
    print(f"Inclusion probabilities:       {pik}")
    print(f"Maximum entropy variance:      {maxent_variance}")
    print(f"Poisson sampling variance:     {poisson_variance}")
    print(
        f"Variance reduction (%):        {100 * (poisson_variance - maxent_variance) / poisson_variance:.2f}%"
    )
    print(f"MaxEnt estimates mean:         {np.mean(maxent_estimates)}")
    print(f"Poisson estimates mean:        {np.mean(poisson_estimates)}")
    print(f"True population total:         {np.sum(population)}")

    # Maximum entropy should have lower variance
    assert (
        maxent_variance < poisson_variance
    ), f"Maximum entropy variance ({maxent_variance}) should be less than Poisson variance ({poisson_variance})"


def test_maxentropy_specific_probability_combinations():
    """
    Test maximum entropy sampling with specific combinations of probabilities
    that might trigger edge cases.
    """
    test_cases = [
        np.array([0.5, 0.5, 0.5, 0.5]),  # Sum = 2, easy to create balanced samples
        np.array([0.9, 0.9, 0.1, 0.1]),  # Strong preference for first two units
        np.array([0.33, 0.33, 0.33]),  # Sum ≈ 1, almost exact for one selection
        np.array([0.1, 0.1, 0.1, 0.1, 0.1]),  # Very sparse selections
    ]

    for pik in test_cases:
        expected_size = int(np.round(np.sum(pik)))

        # Run multiple simulations
        results = []
        for i in range(100):
            np.random.seed(i)
            result = up_max_entropy(pik)
            results.append(result)

            # Check if sample size is as expected
            assert (
                np.sum(result) == expected_size
            ), f"Sample size should be {expected_size} but got {np.sum(result)}"

        # Check that we get different samples (if probability sum > 0)
        if expected_size > 0:
            results_array = np.array(results)
            unique_samples = np.unique(results_array, axis=0)

            # Debug prints for specific probability combinations
            print(f"\nTest case pik: {pik}")
            print(f"Expected sample size: {expected_size}")
            print(f"Number of unique samples observed: {len(unique_samples)}")
            print(f"First few unique samples:")
            for i, sample in enumerate(unique_samples[: min(5, len(unique_samples))]):
                print(f"  Sample {i+1}: {sample}")

            # Should have multiple unique samples unless probabilities force specific outcome
            if len(pik) > expected_size and len(pik) - expected_size > 1:
                assert (
                    len(unique_samples) > 1
                ), f"Should produce different samples for pik={pik}"


def test_maxentropy_exact_n_equals_one_case():
    """
    Explicitly test the n=1 case which uses multinomial sampling.
    """
    # Exactly sum to 1
    pik = np.array([0.2, 0.3, 0.5])

    # Run many simulations
    num_simulations = 5000
    counts = np.zeros(len(pik))

    for i in range(num_simulations):
        np.random.seed(i)
        result = up_max_entropy(pik)

        # Should select exactly one unit
        assert np.sum(result) == 1, "Should select exactly one unit when sum(pik)=1"

        # Count which units get selected
        for j in range(len(pik)):
            if result[j] == 1:
                counts[j] += 1

    # Convert counts to probabilities
    actual_probs = counts / num_simulations

    # Debug prints for n=1 case
    print(f"n=1 test input probabilities: {pik}")
    print(f"n=1 test actual probabilities: {actual_probs}")
    print(f"n=1 test differences: {np.abs(actual_probs - pik)}")
    print(f"n=1 test counts from {num_simulations} simulations: {counts}")

    # Check that selection probabilities match input probabilities
    assert np.allclose(
        actual_probs, pik, atol=0.05
    ), f"Selection probabilities {actual_probs} should match input {pik}"


@pytest.mark.slow
def test_result_sum():
    pik = inclusion_probabilities(np.random.uniform(0, 1, 10000), 500)
    result = up_max_entropy(pik)
    n_result = np.sum(result)
    assert (
        n_result == 500
    ), f"sum of each element of the result was supposed to be 500 but it was {n_result}"
