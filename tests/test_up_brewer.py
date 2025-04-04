import numpy as np
import pytest
from package_sampling import up_brewer


def test_up_brewer_basic():
    pik = np.array([0.0, 0.3, 0.5, 1.0, 0.2])
    selection = up_brewer(pik)

    assert selection.shape == pik.shape
    assert np.all((selection == 0) | (selection == 1))


def test_up_brewer_preserves_zero_one():
    pik = np.array([0.0, 0.7, 0.9, 1.0, 0.5])
    selection = up_brewer(pik)

    assert selection[0] == 0
    assert selection[3] == 1


def test_up_brewer_valid_sample_size():
    pik = np.array([0.1, 0.2, 0.3, 0.4])
    selection = up_brewer(pik)

    assert np.sum(selection) == round(np.sum(pik))


def test_up_brewer_no_missing_values():
    pik = np.array([0.2, 0.5, np.nan, 0.7])
    with pytest.raises(ValueError, match="Missing values detected in the pik vector."):
        up_brewer(pik)


def test_up_brewer_all_outside_range():
    pik = np.array([0.0, 1.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="All elements in pik are outside the range"):
        up_brewer(pik)
