import numpy as np
import pytest

from package_sampling.sampling import up_systematic


def test_up_systematic_basic_sampling():
    pik = np.array([0.1, 0.2, 0.3, 0.4])
    result = up_systematic(pik)

    assert result.shape == pik.shape
    assert np.all((result == 0) | (result == 1))


def test_up_systematic_accepts_list():
    pik = [0.2, 0.5, 0.3]
    result = up_systematic(pik)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.all((result == 0) | (result == 1))


def test_up_systematic_handles_all_zero():
    pik = np.zeros(5)
    result = up_systematic(pik)

    assert np.all(result == 0)


def test_up_systematic_handles_all_one():
    pik = np.ones(4)
    result = up_systematic(pik)

    assert np.all(result == 1)


def test_up_systematic_partial_ones_and_zeros():
    pik = np.array([0.0, 1.0, 0.3, 0.6])
    result = up_systematic(pik)

    assert result[0] == 0
    assert result[1] == 1
    assert np.all((result[2:] == 0) | (result[2:] == 1))


def test_up_systematic_nan_raises_error():
    pik = np.array([0.1, np.nan, 0.3])

    with pytest.raises(ValueError, match="missing values"):
        up_systematic(pik)


def test_up_systematic_determinism_with_seed():
    np.random.seed(123)
    pik = np.array([0.3, 0.3, 0.4])
    r1 = up_systematic(pik)

    np.random.seed(123)
    r2 = up_systematic(pik)

    assert np.array_equal(r1, r2)
