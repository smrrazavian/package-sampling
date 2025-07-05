from __future__ import annotations
import numpy as np
import pytest

from package_sampling.sampling import up_max_entropy
from package_sampling.sampling import up_max_entropy_pi2

from tests.shared_checks import basic_design_checks


# ─────────────────────────────────────────────────────────────
# 1. Core probability sanity
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("N", [90, 210])
def test_maxentropy_with_analytic_pi2(N):
    rng = np.random.default_rng(17)
    pik = rng.uniform(size=N)
    pik *= 40 / pik.sum()
    basic_design_checks(
        up_max_entropy,
        pik,
        fixed_n=True,
        B=800,
        z_threshold=6.0,
        analytic_pi2=up_max_entropy_pi2,
    )


# ─────────────────────────────────────────────────────────────
# 2. Validation / guard-rail behaviour
# ─────────────────────────────────────────────────────────────
def test_maxentropy_coerces_list():
    pik = [0.9, 0.06, 0.04]
    out = up_max_entropy(pik, rng=np.random.default_rng(7))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int8 and out.shape == (3,)


def test_maxentropy_nan_rejected():
    with pytest.raises(ValueError):
        up_max_entropy([0.1, np.nan, 0.4])


def test_maxentropy_out_of_range():
    with pytest.raises(ValueError):
        up_max_entropy([0.2, 1.2, 0.1])


def test_maxentropy_2d_input_rejected():
    with pytest.raises(ValueError):
        up_max_entropy(np.zeros((2, 2)))


# ─────────────────────────────────────────────────────────────
# 3. Branch coverage: n == 0, 1, and deterministic units
# ─────────────────────────────────────────────────────────────
def test_maxentropy_n_equals_zero():
    pik = np.array([1e-7, 5e-7, 2e-7])  # sum ≪ 1 → n == 0
    sel = up_max_entropy(pik)
    assert sel.sum() == 0


def test_maxentropy_n_equals_one():
    pik = np.array([0.2, 0.3, 0.5]) / 1.0  # sum exactly 1 → multinomial path
    rng = np.random.default_rng(99)
    sel = up_max_entropy(pik, rng=rng)
    assert sel.sum() == 1 and sel.dtype == np.int8


def test_maxentropy_deterministic_units():
    pik = np.array([1.0, 0.0, 0.4, 0.6])
    sel = up_max_entropy(pik, rng=np.random.default_rng(5))
    assert sel[0] == 1 and sel[1] == 0
    assert sel.sum() == round(pik.sum())
