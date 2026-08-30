"""Estimators ported from R's *sampling* (ADR-0005).

``HTestimator``, ``varHT``, ``varest`` and ``Hajekestimator`` are all in R and
were all missing here.  ``varHT(method=2)`` is the Sen-Yates-Grundy form the
validation battery needs, so porting them closes a gap and supplies the
instrument at the same time.
"""

from __future__ import annotations

import numpy as np
import pytest

from package_sampling.estimators import (
    design_variance,
    hajek_total,
    ht_total,
    var_deville,
    var_ht,
    var_syg,
)
from package_sampling.sampling import UPMaxEntropy, UPPoisson

PIK = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
Y = np.array([12.0, 4.0, 30.0, 7.0, 19.0])


def _fixed_size_design():
    pik = PIK * 3 / PIK.sum()
    return UPMaxEntropy(pik), pik


# ----------------------------------------------------------------- totals


def test_ht_total_is_the_inverse_probability_sum():
    assert ht_total(Y, PIK) == pytest.approx(np.sum(Y / PIK))


def test_ht_total_is_design_unbiased():
    design, pik = _fixed_size_design()
    rng = np.random.default_rng(0)

    est = np.array(
        [
            ht_total(Y[s == 1], pik[s == 1])
            for s in (design.draw(rng) for _ in range(20_000))
        ]
    )

    se = est.std(ddof=1) / np.sqrt(est.size)
    assert abs(est.mean() - Y.sum()) <= 4 * se


def test_ht_total_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        ht_total(Y, PIK[:3])


def test_hajek_total_divides_by_the_estimated_population_size():
    expected = len(Y) * np.sum(Y / PIK) / np.sum(1.0 / PIK)

    assert hajek_total(Y, PIK, N=len(Y)) == pytest.approx(expected)


# -------------------------------------------------------------- variances


def test_design_variance_matches_the_empirical_variance_fixed_size():
    design, pik = _fixed_size_design()
    rng = np.random.default_rng(1)

    tot = np.array([(design.draw(rng) * Y / pik).sum() for _ in range(50_000)])

    assert design_variance(Y, pik, design.pi2()) == pytest.approx(
        tot.var(), rel=0.05
    )


def test_design_variance_matches_the_empirical_variance_poisson():
    """The double sum is valid for a random-size design too (ADR-0005)."""
    design = UPPoisson(PIK)
    rng = np.random.default_rng(2)

    tot = np.array([(design.draw(rng) * Y / PIK).sum() for _ in range(50_000)])

    assert design_variance(Y, PIK, design.pi2()) == pytest.approx(
        tot.var(), rel=0.05
    )


def test_var_syg_refuses_a_design_with_independent_draws():
    """SYG assumes sum(S_k) = n is constant; for Poisson it silently gives 0."""
    pi2 = UPPoisson(PIK).pi2()

    with pytest.raises(ValueError, match="fixed-size"):
        var_syg(Y, pi2)


def _mean_variance_estimate(estimator, B=20_000, seed=3):
    """Average a SAMPLE variance estimator over B draws of the design."""
    design, pik = _fixed_size_design()
    pi2 = design.pi2()
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(B):
        s = design.draw(rng) == 1
        total += estimator(Y[s], pik[s], pi2[np.ix_(s, s)])
    return total / B, design_variance(Y, pik, pi2)


@pytest.mark.parametrize(
    "name, estimator",
    [
        ("var_ht", lambda y, pik, pi2: var_ht(y, pi2)),
        ("var_syg", lambda y, pik, pi2: var_syg(y, pi2)),
    ],
)
def test_variance_estimators_are_unbiased_over_samples(name, estimator):
    """Both are unbiased for the design variance -- but only from a SAMPLE.

    Fed the whole population they return nonsense (-4088 and 4036 against a
    true variance of 457), because the 1/pi_kl weighting is inverse-probability
    weighting, not part of the variance itself.
    """
    mean_estimate, truth = _mean_variance_estimate(estimator)

    assert mean_estimate == pytest.approx(truth, rel=0.05)


def test_var_ht_rejects_a_non_square_pi2():
    with pytest.raises(ValueError, match="square"):
        var_ht(Y, np.ones((5, 4)))


def test_var_deville_approximates_the_design_variance():
    """varest needs only pik, no pi2 -- the point of Matei & Tille (2005)."""
    mean_estimate, truth = _mean_variance_estimate(
        lambda y, pik, pi2: var_deville(y, pik)
    )

    assert mean_estimate == pytest.approx(truth, rel=0.10)
