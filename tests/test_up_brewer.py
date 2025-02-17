import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


import numpy as np
import pytest
from src.sampling.up_brewer import UPBrewer


@pytest.fixture
def brewer() -> UPBrewer:
    return UPBrewer()


def test_up_brewer_basic(brewer):
    pik = np.array([0.0, 0.3, 0.5, 1.0, 0.2])
    selection = brewer.sample(pik)

    assert selection.shape == pik.shape
    assert np.all((selection == 0) | (selection == 1))


def test_up_brewer_preserves_zero_one(brewer):
    pik = np.array([0.0, 0.7, 0.9, 1.0, 0.5])
    selection = brewer.sample(pik)

    assert selection[0] == 0
    assert selection[3] == 1


def test_up_brewer_valid_sample_size(brewer):
    pik = np.array([0.1, 0.2, 0.3, 0.4])
    selection = brewer.sample(pik)

    assert np.sum(selection) == round(np.sum(pik))


def test_up_brewer_no_missing_values(brewer):
    pik = np.array([0.2, 0.5, np.nan, 0.7])
    with pytest.raises(ValueError, match="Missing values detected in the pik vector."):
        brewer.sample(pik)


def test_up_brewer_all_outside_range(brewer):
    pik = np.array([0.0, 1.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="All elements in pik are outside the range"):
        brewer.sample(pik)
