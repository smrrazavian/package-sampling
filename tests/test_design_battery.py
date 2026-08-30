"""The three-layer validation battery (ADR-0005).

Each instrument is used at the scale where it has power:

1. **Design identities** — exact algebra on the pi2 matrix, no simulation, any
   N. Catches mathematical errors instantly and costs nothing.
2. **Cell-wise pi2** — pi2_hat against the analytic pi2 at small N, where B can
   dominate N^2 and the comparison is unforgiving.
3. **Variance functional** — at high N, where cell-wise is statistically
   hopeless, the whole matrix collapses into one scalar variance per y.

Marked ``slow`` where the cost warrants it; the default run keeps layers 1
and 2.
"""

from __future__ import annotations

import numpy as np
import pytest

from package_sampling.estimators import design_variance, ht_total, var_syg
from package_sampling.sampling import (
    UPBrewer,
    UPMaxEntropy,
    UPPoisson,
    UPSystematic,
    UPTille,
)

# name -> (class, has a closed-form pi2, satisfies the Yates-Grundy condition)
DESIGNS = {
    "UPBrewer": (UPBrewer, False, True),
    "UPTille": (UPTille, True, True),
    "UPSystematic": (UPSystematic, True, False),  # famously violates YG
    "UPMaxEntropy": (UPMaxEntropy, True, True),
    "UPPoisson": (UPPoisson, True, True),
}
WITH_PI2 = [n for n, (_, has, _) in DESIGNS.items() if has]


def make_pik(N, n, seed=0):
    rng = np.random.default_rng(seed)
    pik = rng.uniform(0.05, 1.0, size=N)
    pik *= n / pik.sum()
    return np.clip(pik, 1e-9, 1.0 - 1e-9)


# =====================================================================
# Layer 1 — design identities. No simulation, valid at any N.
# =====================================================================
@pytest.mark.parametrize("name", WITH_PI2)
@pytest.mark.parametrize("N,n", [(40, 12), (300, 40)])
def test_pi2_is_symmetric_with_pik_on_the_diagonal(name, N, n):
    cls, _, _ = DESIGNS[name]
    pik = make_pik(N, n)

    pi2 = cls(pik).pi2()

    np.testing.assert_allclose(pi2, pi2.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(pi2), pik, atol=1e-9)


@pytest.mark.parametrize("name", WITH_PI2)
@pytest.mark.parametrize("N,n", [(40, 12), (300, 40)])
def test_fixed_size_designs_satisfy_the_row_sum_identity(name, N, n):
    """sum_l pi_kl = n * pi_k holds only for a fixed sample size."""
    cls, _, _ = DESIGNS[name]
    pik = make_pik(N, n)
    design = cls(pik)
    pi2 = design.pi2()

    row = pi2.sum(axis=1)

    if design.fixed_size:
        np.testing.assert_allclose(row, design.n * pik, rtol=1e-6, atol=1e-7)
    else:
        # Poisson: sum_l pi_kl = pi_k * (sum(pik) + 1 - pi_k)
        np.testing.assert_allclose(
            row, pik * (pik.sum() + 1.0 - pik), rtol=1e-12
        )


@pytest.mark.parametrize("name", WITH_PI2)
def test_yates_grundy_condition_holds_exactly_where_theory_says(name):
    """pi_kl <= pi_k pi_l guarantees a non-negative SYG variance.

    Systematic sampling does not satisfy it -- that is a property of the
    design, which is why SYG estimates can go negative under it, and asserting
    it universally would assert something false.
    """
    cls, _, satisfies_yg = DESIGNS[name]
    pik = make_pik(80, 20)

    pi2 = cls(pik).pi2()
    off = ~np.eye(pik.size, dtype=bool)
    worst = (pi2 - np.outer(pik, pik))[off].max()

    if satisfies_yg:
        assert worst <= 1e-9, f"{name} unexpectedly violates Yates-Grundy"
    else:
        assert worst > 1e-6, f"{name} was expected to violate Yates-Grundy"


@pytest.mark.parametrize("name", WITH_PI2)
def test_pi2_respects_the_frechet_upper_bound(name):
    """pi_kl <= min(pi_k, pi_l): a pair cannot co-occur more than either unit."""
    cls, _, _ = DESIGNS[name]
    pik = make_pik(80, 20)

    pi2 = cls(pik).pi2()

    assert np.all(pi2 <= np.minimum.outer(pik, pik) + 1e-9)


@pytest.mark.parametrize("name", list(DESIGNS))
def test_fixed_size_designs_realise_a_constant_sample_size(name):
    cls, _, _ = DESIGNS[name]
    pik = make_pik(60, 15)
    design = cls(pik)
    rng = np.random.default_rng(4)

    sizes = {int(design.draw(rng).sum()) for _ in range(300)}

    if design.fixed_size:
        assert sizes == {design.n}
    else:
        assert len(sizes) > 1, "Poisson sample size should vary"


# =====================================================================
# Layer 2 — cell-wise pi2_hat vs the analytic pi2, small N only.
# =====================================================================
@pytest.mark.parametrize("name", WITH_PI2)
def test_empirical_pi2_matches_the_analytic_matrix_at_small_n(name):
    """Needs B >> N^2, so it runs only where that is affordable."""
    cls, _, _ = DESIGNS[name]
    N, B = 12, 60_000
    pik = make_pik(N, 4, seed=11)
    design = cls(pik)
    rng = np.random.default_rng(5)

    acc = np.zeros((N, N))
    for _ in range(B):
        s = design.draw(rng).astype(float)
        acc += np.outer(s, s)
    hat = acc / B

    analytic = design.pi2()
    se = np.sqrt(np.clip(analytic * (1 - analytic), 0, None) / B)
    assert np.all(np.abs(hat - analytic) <= 5 * se + 1e-3), (
        f"{name}: max deviation "
        f"{np.max(np.abs(hat - analytic) - 5 * se):.4f} beyond 5 SE"
    )


# =====================================================================
# Layer 3 — variance functional at high N.
# =====================================================================
def _y_vectors(pik, rng):
    """A spanning set: the easy case, the hard case, and noise."""
    # y proportional to pik is handled separately: it makes the HT estimator
    # exactly constant, so its variance is zero and no relative tolerance can
    # describe the comparison.
    return {
        "orthogonal-ish to pik": (pik - pik.mean()) * 100.0,
        "random": rng.uniform(1, 100, size=pik.size),
        "adversarial spike": np.where(
            np.arange(pik.size) == int(np.argmin(pik)), 1000.0, 1.0
        ),
    }


@pytest.mark.slow
@pytest.mark.parametrize("name", WITH_PI2)
def test_design_variance_matches_the_empirical_variance(name):
    cls, _, _ = DESIGNS[name]
    N, n, B = 200, 40, 20_000
    pik = make_pik(N, n, seed=7)
    design = cls(pik)
    pi2 = design.pi2()
    rng = np.random.default_rng(9)

    draws = np.array([design.draw(rng) for _ in range(B)], dtype=float)

    for label, y in _y_vectors(pik, np.random.default_rng(3)).items():
        empirical = (draws * (y / pik)).sum(axis=1).var()
        exact = design_variance(y, pik, pi2)
        se = empirical * np.sqrt(2.0 / B)  # SE of a variance estimate
        assert abs(empirical - exact) <= 6 * se + 1e-6 * exact, (
            f"{name} / {label}: empirical {empirical:.4f} vs exact "
            f"{exact:.4f}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("name", [n for n in WITH_PI2 if n != "UPPoisson"])
def test_y_proportional_to_pik_gives_exactly_zero_variance(name):
    """The classical degenerate case, and a sharp test of the pi2 matrix.

    Under a fixed-size design with ``y = c * pik`` the estimator collapses to
    ``sum(S_k) * c = n * c``, a constant, so ``Var(Y_HT) = 0`` exactly. The
    double sum must therefore cancel to nothing; what survives is pure
    floating-point residue, which is only meaningful against the size of the
    terms being cancelled.
    """
    cls, _, _ = DESIGNS[name]
    N, n, B = 120, 30, 3_000
    pik = make_pik(N, n, seed=7)
    design = cls(pik)
    pi2 = design.pi2()
    y = pik * 100.0
    rng = np.random.default_rng(9)

    realised = np.array(
        [(design.draw(rng) * (y / pik)).sum() for _ in range(B)]
    )
    assert realised.var() == 0.0, "Y_HT must be constant when y is prop to pik"

    # With y = c*pik the double sum collapses to c^2 * sum_k(row sum of Delta),
    # and every row sum is zero for a fixed-size design. So whatever survives
    # is exactly the pi2 matrix's own inaccuracy, and that is the right bound
    # to measure against -- not machine epsilon.
    #
    # It matters which design: UPTille and UPSystematic build pi2 to within an
    # ulp, while UPMaxEntropy is limited by upme_pik_tilde_from_pik's eps=1e-6
    # fixed point, leaving row-sum residuals around 1e-8.
    row_residual = np.abs(pi2.sum(axis=1) - design.n * pik).sum()
    bound = 100.0**2 * row_residual + 1e-9 * np.abs(y).sum()

    assert abs(design_variance(y, pik, pi2)) <= bound


@pytest.mark.slow
@pytest.mark.parametrize("name", list(DESIGNS))
def test_horvitz_thompson_is_unbiased_at_high_n(name):
    cls, _, _ = DESIGNS[name]
    N, n, B = 200, 40, 20_000
    pik = make_pik(N, n, seed=7)
    design = cls(pik)
    rng = np.random.default_rng(13)
    y = np.random.default_rng(2).uniform(1, 100, size=N)

    est = np.array(
        [
            ht_total(y[s == 1], pik[s == 1])
            for s in (design.draw(rng) for _ in range(B))
        ]
    )

    se = est.std(ddof=1) / np.sqrt(B)
    assert abs(est.mean() - y.sum()) <= 5 * se


@pytest.mark.slow
def test_syg_estimator_is_unbiased_for_a_fixed_size_design():
    """The SYG estimator, averaged over samples, recovers the design variance."""
    N, n, B = 60, 15, 20_000
    pik = make_pik(N, n, seed=21)
    design = UPMaxEntropy(pik)
    pi2 = design.pi2()
    y = np.random.default_rng(6).uniform(1, 100, size=N)
    rng = np.random.default_rng(8)

    total = 0.0
    for _ in range(B):
        s = design.draw(rng) == 1
        total += var_syg(y[s], pi2[np.ix_(s, s)])

    assert total / B == pytest.approx(design_variance(y, pik, pi2), rel=0.05)
