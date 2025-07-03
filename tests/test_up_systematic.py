import numpy as np
import pytest

from package_sampling.sampling import up_systematic, up_systematic_pi2
from tests.shared_checks import basic_design_checks


# ----------------------------------------------------------------------
# 1. Probability & π₂ sanity (fast)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("N", [60, 180])
def test_systematic_probabilities(N):
    rng = np.random.default_rng(123)
    pik = rng.uniform(size=N)
    pik *= 25 / pik.sum()

    basic_design_checks(
        up_systematic,
        pik,
        fixed_n=False,
        B=1_000,
        analytic_pi2=up_systematic_pi2,
    )


# ----------------------------------------------------------------------
# 2. Guard-rail behaviour identical to earlier ad-hoc tests
# ----------------------------------------------------------------------
def test_systematic_edge_cases():
    # 0’s and 1’s preserved
    pik = np.array([0.0, 1.0, 0.4, 0.6])
    sel = up_systematic(pik, rng=np.random.default_rng(5))
    assert sel[0] == 0 and sel[1] == 1
    assert sel.dtype == np.int8

    with pytest.raises(ValueError):
        up_systematic([0.2, np.nan, 0.5])


# ----------------------------------------------------------------------
# 3.  Guard-rails for up_systematic_pi2
# ----------------------------------------------------------------------
def test_pi2_accepts_list_and_coerces():
    pik = [0.2, 0.5, 0.3]
    pi2 = up_systematic_pi2(pik)
    assert isinstance(pi2, np.ndarray)
    np.testing.assert_allclose(np.diag(pi2), pik, atol=1e-12)


def test_pi2_all_deterministic():
    # No live units (all 0 or 1) → branch N == 0 → outer product
    pik = np.array([0.0, 1.0, 1.0, 0.0])
    pi2 = up_systematic_pi2(pik)
    np.testing.assert_allclose(pi2, np.outer(pik, pik), atol=1e-12)
