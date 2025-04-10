"""
Utility functions for the package_sampling module.

This includes:
- `as_int`: Safely coerce floats to integers with optional validation
- `inclusion_probabilities`: Compute inclusion probabilities
    for a given vector and sample size
"""

from .as_int import as_int  # noqa: F401
from .inclusion_probabilities import inclusion_probabilities  # noqa: F401

__all__ = [
    "as_int",
    "inclusion_probabilities",
]
