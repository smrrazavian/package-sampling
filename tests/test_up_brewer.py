import numpy as np
import pytest

from package_sampling.sampling import up_brewer
from tests.shared_checks import basic_design_checks


# ----------------------------------------------------------------------
# 1. Probability sanity with Monte-Carlo (analytic π₂ unavailable)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("N", [50, 150])
def test_brewer_probabilities(N):
    rng = np.random.default_rng(7)
    pik = rng.uniform(size=N)
    pik *= 20 / pik.sum()

    basic_design_checks(
        up_brewer,
        pik,
        fixed_n=True,
        B=1_000,
    )


# ----------------------------------------------------------------------
# 2. Guard-rail tests (previous ones kept, dtype now int8)
# ----------------------------------------------------------------------
def test_brewer_edge_cases():
    pik = np.array([0.0, 1.0, 0.4, 0.6])
    sel = up_brewer(pik, rng=np.random.default_rng(3))
    assert sel[0] == 0 and sel[1] == 1
    assert sel.dtype == np.int8

    with pytest.raises(ValueError):
        up_brewer([0.2, np.nan, 0.5])

    with pytest.raises(ValueError):
        up_brewer([0.0, 1.0, 1.0, 0.0])
