# Package Sampling

A Python implementation of various probability-based sampling algorithms, inspired by the R package [sampling](https://cran.r-project.org/web/packages/sampling/index.html). This package offers a variety of sampling methods like `Tillé's Method`, `Poisson Sampling`, `Systematic Sampling`, and more, designed for **unequal probability** sampling. The algorithms are implemented in a way that supports both theoretical understanding and real-world use cases.

## Installation

You can install the package directly from PyPI:

```bash
pip install package-sampling
```

## Usage Example

```python
from package_sampling.sampling import up_brewer
import numpy as np

probabilities = np.array([0.1, 0.2, 0.3, 0.4])

# Draw a sample using Brewer's method
sample = up_brewer(probabilities)
print(sample)
```

## Authors

Mohammadreza Razavian - <smrrazavian@outlook.com>
Bardia Panahbehagh - <Panahbehagh@khu.ac.ir>

## Citation

If you use this package in your work, please cite it as follows:

### APA Style

`Razavian, M., & Panahbehagh, B. (2025). Package Sampling: A Python implementation of various probability-based sampling algorithms. https://github.com/smrrazavian/package-sampling.`

### BibTex

```bibtex
@misc{razavian2025packagesampling,
  author = {Razavian, Mohammadreza and Panahbehagh, Bardia},
  title = {Package Sampling: A Python implementation of various probability-based sampling algorithms},
  year = {2025},
  url = {https://github.com/smrrazavian/package-sampling}
}
```

## Acknowledgments

This package is inspired by the R package "Sampling"
Thanks to all contributors and users

## Contributing

We welcome contributions to the package! If you have suggestions for new algorithms, improvements, or bug fixes, feel free to fork the repository and submit a pull request. Please ensure that your code adheres to the existing style and includes tests for any new functionality.
Steps to Contribute:

1. Fork the repository.
2. Clone your fork to your local machine.
3. Create a new branch for your changes.
4. Make your changes and commit them with clear messages.
5. Push your changes to your fork.
6. Open a pull request describing your changes.
