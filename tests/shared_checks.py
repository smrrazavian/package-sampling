from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


# ----------------------------------------------------------------------
# Monte-Carlo helpers
# ----------------------------------------------------------------------
def mc_estimates(
    design_fun,
    pik: NDArray,
    B: int = 1_000,
    seed: int = 1,
) -> tuple[NDArray, NDArray]:
    """Empirical π̂₁ and π̂₂ from *B* replicates."""
    rng = np.random.default_rng(seed)
    N = pik.size
    phat = np.zeros(N)
    phat2 = np.zeros((N, N))
    for _ in range(B):
        mask = design_fun(pik, rng=rng)
        phat += mask
        phat2 += np.outer(mask, mask)
    phat /= B
    phat2 /= B
    np.fill_diagonal(phat2, phat)  # self-consistency
    return phat, phat2


# ----------------------------------------------------------------------
# Generic probability & consistency battery
# ----------------------------------------------------------------------
def basic_design_checks(
    design_fun,
    pik: NDArray,
    *,
    fixed_n: bool,
    B: int = 1_000,
    z_threshold: float = 4.0,
    analytic_pi2=None,
) -> None:
    """
    design_fun   : callable(pik, rng=Generator) → 0/1 mask
    pik          : first-order inclusion probabilities
    fixed_n      : True → assert realised n == round(sum(pik))
    B            : number of Monte-Carlo replicates
    z_threshold  : statistical tolerance for first-order π̂₁
    analytic_pi2 : callable(pik) → π₂ matrix (skip MC) or None
    """
    mask = design_fun(pik, rng=np.random.default_rng(42))
    assert mask.dtype == np.int8 and mask.shape == pik.shape
    if fixed_n:
        assert mask.sum() == round(pik.sum())

    # ---------- first-order ----------
    p1_hat, _ = mc_estimates(design_fun, pik, B=B)
    sigma1 = np.sqrt(pik * (1 - pik) / B)
    assert np.all(
        np.abs(p1_hat - pik) <= z_threshold * sigma1
    ), "π̂ outside statistical tolerance"

    # ---------- second-order ----------
    if analytic_pi2 is not None:
        p2 = analytic_pi2(pik)
        # analytic result → diagonals must equal target π
        np.testing.assert_allclose(np.diag(p2), pik, atol=1e-12)

        # strict upper bound: π_ij ≤ min(π_i, π_j)
        lim = np.minimum.outer(pik, pik) + 1e-12
        assert np.all(p2 <= lim), "π₂ violates upper bound"
    else:
        # Monte-Carlo estimate
        _, p2 = mc_estimates(design_fun, pik, B=B)
        np.testing.assert_allclose(np.diag(p2), p1_hat, atol=1e-12)

        # allow sampling error on upper-bound check
        lim = np.minimum.outer(pik, pik)
        sigma2 = np.sqrt(p2 * (1 - p2) / B)  # rough SE for each cell
        assert np.all(
            p2 <= lim + z_threshold * sigma2 + 1e-12
        ), "π₂ exceeds upper bound beyond MC tolerance"
