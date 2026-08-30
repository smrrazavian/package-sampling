"""``as_int`` rejects genuinely non-integer input but still absorbs float noise.

R's ``.as_int`` warns and returns the value *unrounded*, which lets a
fractional sample size flow into loop bounds.  We reject it instead
(ADR-0007).
"""

from __future__ import annotations

import numpy as np
import pytest

from package_sampling.utils import as_int


def test_rejects_a_genuinely_non_integer_value():
    with pytest.raises(ValueError, match="not an integer"):
        as_int(2.02)


def test_absorbs_floating_point_noise():
    """sum(pik) rarely lands exactly on an integer; that must still work."""
    noisy = 24.999999999999996

    assert as_int(noisy) == 25


def test_passes_through_exact_integers():
    assert as_int(7) == 7
    assert as_int(7.0) == 7
    assert as_int(np.int64(7)) == 7


def test_tolerance_matches_r_all_equal():
    """R's .as_int uses all.equal, tolerance sqrt(machine eps) = 1.49e-08.

    Ours was np.isclose(atol=1e-6), whose default rtol=1e-5 accepts a
    relative error ~670x looser -- at sum 200 it let 200.0015 through, which
    R rejects (ADR-0007).
    """
    with pytest.raises(ValueError, match="not an integer"):
        as_int(200.0015)

    # measured float error in sum(pik) is ~2e-16 relative even at N=1e6,
    # so a legitimate sum must still pass with room to spare
    assert as_int(200.0 + 200.0 * 1e-12) == 200


def test_near_zero_uses_an_absolute_tolerance():
    """all.equal compares absolutely when the rounded target is zero."""
    assert as_int(1e-9) == 0

    with pytest.raises(ValueError, match="not an integer"):
        as_int(1.000005e-6)
