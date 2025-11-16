# Package Sampling

Python implementations of unequal-probability sampling algorithms inspired by
the R package [sampling](https://cran.r-project.org/web/packages/sampling/index.html).
The library focuses on survey-sampling designs where each unit can have its own
inclusion probability.  Implementations favour clarity, deterministic edge-case
handling, and full property tests so they can be used both in teaching and in
production pipelines.

## Features

- Brewer, Tillé, Systematic, Poisson, and Max-Entropy samplers, plus helper
  transforms (`q-from-w`, `π̃-from-π`, joint π₂ blocks, etc.).
- Deterministic handling of units with π≈0 or π≈1 and reproducible RNG support.
- 100% coverage tests for the Poisson/Systematic routines and statistical
  checks for the remaining methods.
- Typed NumPy APIs and lightweight `numpy` dependency only.

## Installation

Install from PyPI:

```bash
pip install package-sampling
```

## Usage

```python
import numpy as np
from package_sampling.sampling import (
    up_brewer,
    up_poisson,
    up_systematic,
)

pik = np.array([0.1, 0.2, 0.3, 0.4])

# Draw a fixed-size sample using Brewer’s method
brewer_sample = up_brewer(pik)

# Draw a Poisson sample (random size)
poisson_sample = up_poisson(pik)

# Deterministic handling of π≈0 or π≈1, RNG optional
systematic_sample = up_systematic(pik, rng=np.random.default_rng(123))
```

The `package_sampling.sampling` module exposes every design via `__all__`, so
you can also `from package_sampling import sampling as samp` and work with
`samp.up_tille`, `samp.upme_q_from_w`, etc.

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

## Contributing

We welcome contributions!  Please ensure any change preserves the statistical
properties of the designs and includes regression tests.

**Development workflow**

```bash
git clone https://github.com/smrrazavian/package-sampling.git
cd package-sampling
poetry install
poetry run pytest   # or `make precommit`
```

Before opening a PR, run `poetry run black package_sampling tests`,
`poetry run isort ...`, `poetry run flake8 ...`, and `poetry run mypy package_sampling`
to keep formatting and typing consistent.  The `Makefile` exposes convenience
targets (`make format`, `make lint`, `make test`, etc.) if you prefer.

**Steps to contribute**

1. Fork the repository.
2. Clone your fork to your local machine.
3. Create a new branch for your changes.
4. Make your changes and commit them with clear messages.
5. Push your changes to your fork.
6. Open a pull request describing your changes.

## Acknowledgments

- Inspired by the R package **sampling** (Tillé et al.).
- Thanks to everyone who opened issues, contributed code, or verified the
  statistical behavior against external benchmarks.

## License

Licensed under the GNU General Public License v3.0 (or any later version).
See [`LICENSE`](LICENSE) for the full text.  Because the project ports
algorithms from the R **sampling** package (GPL ≥ 2), keeping this project
GPL-compatible ensures downstream users remain compliant.
