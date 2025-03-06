import numpy as np
from typing import List, Union
from package_sampling.base import SamplingMethod
from package_sampling.utils import inclusion_probabilities
from package_sampling.utils import as_int


class UPTille(SamplingMethod):
    """
    Implements Tillé's method for Unequal Probability (UP) sampling.

    This algorithm selects a fixed number of elements from a population
    while maintaining the predefined inclusion probabilities.

    Mathematics of the UPTille:
    - Let `pik` be the inclusion probability of the i-th element.
    - The algorithm selects a sample of size `n` from the population.
    - It ensures that the selection respects the inclusion probabilities.
    - The algorithm uses an iterative process to adjust the selection vector.
    """

    def sample(
        self, pik: Union[List[float], np.ndarray], eps: float = 1e-6
    ) -> np.ndarray:
        """
        Selects a sample using Tillé's method for Unequal Probability sampling.

        Args:
            pik (Union[List[float], np.ndarray]): A 1D list or NumPy array representing inclusion probabilities.
                Values should be in the range (0, 1).
            eps (float, optional): A small threshold to handle floating-point precision issues. Defaults to 1e-6.

        Returns:
            np.ndarray: A 1D array of the same shape as `pik`, containing only 0s and 1s.
            A `1` indicates that the corresponding element was selected in the sample.

        Raises:
            ValueError: If `pik` contains NaN values.
            ValueError: If all elements of `pik` are outside the range [eps, 1-eps].
        """
        if not isinstance(pik, np.ndarray):
            pik = np.array(pik)

        # Handle empty input case
        if pik.size == 0:
            return np.array([], dtype=int)

        if np.any(np.isnan(pik)):
            raise ValueError("There are missing values in the `pik` vector.")

        eligible_mask = (pik > eps) & (pik < 1 - eps)
        pikb = pik[eligible_mask]
        N = pikb.size

        if N < 1:
            raise ValueError(
                "All elements in `pik` are outside the range [eps, 1-eps]."
            )

        s = np.zeros_like(pik, dtype=int)
        selection_vector = np.ones(N, dtype=int)
        previous_inclusion_probs = np.ones(N)
        n = as_int(np.round(np.sum(pikb)))

        # Check if n is valid (between 0 and N)
        if n <= 0:
            return np.zeros_like(pik, dtype=int)
        if n >= N:
            s[eligible_mask] = 1
            return s

        for i in range(max(1, N - n)):
            a = inclusion_probabilities(pikb, N - i)
            # Avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                adjusted_probs = np.where(
                    previous_inclusion_probs > 0, 1 - a / previous_inclusion_probs, 0
                )
            previous_inclusion_probs = a
            p = adjusted_probs * selection_vector

            # Avoid division by zero in normalization
            if np.sum(p) > 0:
                p = np.cumsum(p / np.sum(p))
                u = np.random.uniform()
                if u < p[-1]:
                    selection_vector[np.searchsorted(p, u)] = 0
                else:
                    selection_vector[-1] = 0
            else:
                # If all probabilities are zero, randomly select an element to remove
                idx = np.random.choice(np.where(selection_vector == 1)[0])
                selection_vector[idx] = 0

        s[eligible_mask] = selection_vector
        return s
