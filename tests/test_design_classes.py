"""Design classes: build once from pik, draw many times (ADR-0002)."""

from __future__ import annotations

import numpy as np
import pytest

from package_sampling.sampling import (
    UPPoisson,
    UPBrewer,
    UPMaxEntropy,
    UPSystematic,
    UPTille,
    up_brewer,
    up_max_entropy,
    up_poisson,
    up_tille,
    up_systematic,
)


def test_poisson_draw_returns_int8_mask():
    pik = np.array([0.2, 0.5, 0.7, 0.9])

    s = UPPoisson(pik).draw(np.random.default_rng(0))

    assert s.dtype == np.int8
    assert s.shape == pik.shape
    assert np.all(np.isin(s, [0, 1]))


def test_up_poisson_accepts_rng_and_is_reproducible():
    pik = np.array([0.2, 0.5, 0.7, 0.9])

    a = up_poisson(pik, rng=np.random.default_rng(1))
    b = up_poisson(pik, rng=np.random.default_rng(1))

    assert np.array_equal(a, b)
    assert a.dtype == np.int8


def test_poisson_pi2_is_outer_product_with_pik_on_diagonal():
    pik = np.array([0.2, 0.5, 0.7, 0.9])

    p2 = UPPoisson(pik).pi2()

    expected = np.outer(pik, pik)
    np.fill_diagonal(expected, pik)
    np.testing.assert_allclose(p2, expected, atol=1e-15)


def test_systematic_class_and_one_shot_agree_given_the_same_rng():
    rng0 = np.random.default_rng(4)
    pik = rng0.uniform(size=40)
    pik *= 12 / pik.sum()

    from_class = UPSystematic(pik).draw(np.random.default_rng(9))
    from_function = up_systematic(pik, rng=np.random.default_rng(9))

    np.testing.assert_array_equal(from_class, from_function)


def test_max_entropy_class_and_one_shot_agree_given_the_same_rng():
    rng0 = np.random.default_rng(11)
    pik = rng0.uniform(size=30)
    pik *= 10 / pik.sum()

    from_class = UPMaxEntropy(pik).draw(np.random.default_rng(3))
    from_function = up_max_entropy(pik, rng=np.random.default_rng(3))

    np.testing.assert_array_equal(from_class, from_function)


def test_max_entropy_design_is_built_once_and_reused_across_draws():
    rng0 = np.random.default_rng(11)
    pik = rng0.uniform(size=30)
    pik *= 10 / pik.sum()

    design = UPMaxEntropy(pik)
    q_first = design.q

    draws = [design.draw(np.random.default_rng(s)) for s in range(5)]

    assert design.q is q_first, "q must not be recomputed per draw"
    assert all(d.sum() == design.n for d in draws)


def test_tille_class_and_one_shot_agree_given_the_same_rng():
    rng0 = np.random.default_rng(21)
    pik = rng0.uniform(size=25)
    pik *= 8 / pik.sum()

    from_class = UPTille(pik).draw(np.random.default_rng(6))
    from_function = up_tille(pik, rng=np.random.default_rng(6))

    np.testing.assert_array_equal(from_class, from_function)


def test_tille_elimination_probabilities_are_built_once():
    rng0 = np.random.default_rng(21)
    pik = rng0.uniform(size=25)
    pik *= 8 / pik.sum()

    design = UPTille(pik)
    v_first = design.v

    for seed in range(5):
        design.draw(np.random.default_rng(seed))

    assert design.v is v_first, "v must not be recomputed per draw"


def test_brewer_class_and_one_shot_agree_given_the_same_rng():
    rng0 = np.random.default_rng(31)
    pik = rng0.uniform(size=25)
    pik *= 8 / pik.sum()

    from_class = UPBrewer(pik).draw(np.random.default_rng(2))
    from_function = up_brewer(pik, rng=np.random.default_rng(2))

    np.testing.assert_array_equal(from_class, from_function)


def test_brewer_pi2_refuses_because_no_closed_form_exists():
    pik = np.array([0.2, 0.3, 0.5, 0.4, 0.6])

    with pytest.raises(NotImplementedError, match="closed form"):
        UPBrewer(pik).pi2()


@pytest.mark.parametrize(
    "cls, expected",
    [
        (UPBrewer, True),
        (UPMaxEntropy, True),
        (UPSystematic, True),
        (UPTille, True),
        (UPPoisson, False),
    ],
)
def test_designs_declare_whether_they_are_fixed_size(cls, expected):
    """Sen-Yates-Grundy is only valid for fixed-size designs (ADR-0005)."""
    # sum(pik) must be an integer: max-entropy's pi-tilde fixed point cannot
    # converge otherwise, which is the case ADR-0007 turns into a clear error.
    pik = np.array([0.2, 0.3, 0.5, 0.4, 0.6])

    assert cls(pik).fixed_size is expected
