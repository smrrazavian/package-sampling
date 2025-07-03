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


def test_tille_n_equals_zero():
    pik = np.array([1e-5, 2e-5, 3e-5])  # sum ≈ 0 → n == 0
    out = up_tille(pik)
    assert np.all(out == 0)


def test_tille_n_equals_N():
    # All live units must be selected when n == N
    pik = np.array([0.6, 0.7, 0.8])
    pik *= 3 / pik.sum()  # force n == N
    out = up_tille(pik)
    assert np.all(out == 1)


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


def test_pi2_full_census():
    pik = np.array([0.5, 0.7, 0.9])
    pik *= 3 / pik.sum()  # n == N
    pi2 = up_tille_pi2(pik)
    assert np.all(pi2 == 1.0)


def test_pi2_off_diagonal_consistency():
    pik = np.array([0.2, 0.5, 0.3])
    pi2 = up_tille_pi2(pik)
    lim = np.minimum.outer(pik, pik) + 1e-12
    assert np.all(pi2 <= lim)
