# Package Sampling

python version of Package Sampling in R

## Installation

You can install the package directly from PyPI:

```bash
pip install package-sampling
```

## Usage Example

```python
from package_sampling import up_brewer

# Create a population with inclusion probabilities
probabilities = [0.1, 0.2, 0.3, 0.4]

# Draw a sample using Brewer's method
sample = up_brewer(probabilities)
print(sample)
```

## Authors

Mohammadreza Razavian - <smrrazavian@outlook.com>
Bardia Panahbehagh - <Panahbehagh@khu.ac.ir>

## Acknowledgments

This package is inspired by the R package "Sampling"
Thanks to all contributors and users
