import numpy as np
import pytest

from package_sampling.sampling import up_tille, up_tille_pi2
from tests.shared_checks import basic_design_checks


# ==========================================================================
# 1.  Core probability tests for the fixed-size design
# ==========================================================================
@pytest.mark.parametrize("N", [73, 211])
def test_tille_probabilities(N):
    rng = np.random.default_rng(0)
    pik = rng.uniform(size=N)
    pik *= 30 / pik.sum()  # expected n ≈ 30

    basic_design_checks(
        up_tille,
        pik,
        fixed_n=True,
        B=1_000,
        analytic_pi2=up_tille_pi2,  # exact π₂ → fast
    )


# ==========================================================================
# 2.  Validation / guard-rail tests for up_tille
# ==========================================================================
def test_tille_coerces_list():
    pik = [0.2, 0.3, 0.5]
    out = up_tille(pik, rng=np.random.default_rng(7))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int8 and out.shape == (3,)


def test_tille_nan_rejected():
    pik = [0.1, np.nan, 0.4]
    with pytest.raises(ValueError, match="NaN"):
        up_tille(pik)


@pytest.mark.parametrize(
    "pik",
    [
        [1e-8, 2e-8, 5e-8],  # near-zero
        [0.9999999, 0.9999998],  # near-one
    ],
)
def test_tille_all_outside_range(pik):
    with pytest.raises(ValueError, match="outside"):
        up_tille(pik)


def test_tille_rejects_a_near_zero_non_integer_sum():
    """sum(pik_live) = 6e-05 is not a valid fixed-size sample size.

    This used to assert an all-zero sample.  R disagrees: it loops
    ``1:(3 - 6e-05)`` = ``c(1, 2)``, removes two units and returns a sample of
    size 1.  Rather than pick between an artifact and an accident we reject
    the input (ADR-0007).
    """
    pik = np.array([1e-5, 2e-5, 3e-5])

    with pytest.raises(ValueError, match="not an integer"):
        up_tille(pik)


def test_tille_n_equals_zero_selects_nobody():
    """n == 0 needs a live unit whose sum still rounds to zero.

    as_int's tolerance is tau = sqrt(machine eps) = 1.49e-08, and "live" means
    pi > eps, so the branch is reachable only when the caller sets eps below
    tau.  At the default eps = 1e-6 > tau it cannot be reached at all.
    """
    pik = np.array([1e-9])

    out = up_tille(pik, eps=1e-12, rng=np.random.default_rng(0))

    assert out.sum() == 0
    assert out.dtype == np.int8 and out.shape == pik.shape


def test_pi2_n_equals_zero_is_the_diagonal():
    pik = np.array([1e-9])

    pi2 = up_tille_pi2(pik, eps=1e-12)

    np.testing.assert_allclose(np.diag(pi2), pik, atol=1e-15)


def test_tille_zero_total_branch(monkeypatch):
    """
    Force v * sb to be all-zero so that `total == 0` and the algorithm
    falls back to rng.choice(np.flatnonzero(sb)).
    """
    import importlib

    tille_mod = importlib.import_module("package_sampling.sampling.up_tille")

    # fake inclusion_probabilities → returns vector identical to prev_b
    def fake_incprob(pik_live, m):
        return np.ones_like(pik_live)  # makes v = 0

    monkeypatch.setattr(
        tille_mod, "inclusion_probabilities", fake_incprob, raising=True
    )

    pik = np.array([0.2, 0.3, 0.5])  # sum ≈ 1 ⇒ n = 1, N = 3
    out = tille_mod.up_tille(pik, rng=np.random.default_rng(99))

    # Branch hit ⇒ only *one* unit eliminated ⇒ sample size = 1
    assert out.sum() == 1
    assert out.dtype == np.int8 and out.shape == pik.shape


# ==========================================================================
# 3.  Guard-rail tests for up_tille_pi2 (previously uncovered)
# ==========================================================================
def test_pi2_coerces_list():
    pik = [0.3, 0.4, 0.3]
    pi2 = up_tille_pi2(pik)
    assert isinstance(pi2, np.ndarray)
    np.testing.assert_allclose(np.diag(pi2), pik, atol=1e-12)


@pytest.mark.parametrize(
    "pik",
    [
        [1e-8, 2e-8],  # effectively zeros
        [0.9999999, 0.9999998],  # effectively ones
    ],
)
def test_pi2_all_outside_range(pik):
    with pytest.raises(ValueError, match="outside"):
        up_tille_pi2(pik)


def test_pi2_off_diagonal_consistency():
    pik = np.array([0.2, 0.5, 0.3])
    pi2 = up_tille_pi2(pik)
    lim = np.minimum.outer(pik, pik) + 1e-12
    assert np.all(pi2 <= lim)
