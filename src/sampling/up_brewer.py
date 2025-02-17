import numpy as np
from src.base import SamplingMethod
from src.utils.as_int import as_int


class UPBrewer(SamplingMethod):
    def sample(self, pik: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        if np.any(np.isnan(pik)):
            raise ValueError("Missing values detected in the pik vector.")

        eligible_mask = (pik > eps) & (pik < 1 - eps)
        pikb = pik[eligible_mask]

        if pikb.size == 0:
            raise ValueError("All elements in pik are outside the range [eps, 1-eps].")

        result = np.copy(pik)
        sb = np.zeros_like(pikb, dtype=int)
        total_pikb = as_int(np.sum(pikb))

        for i in range(total_pikb):
            a = np.dot(pikb, sb)
            denom = (total_pikb - a) - pikb * (total_pikb - i + 1)

            denom = np.where(np.abs(denom) < eps, eps, denom)

            p = (1 - sb) * pikb * ((total_pikb - a) - pikb) / denom
            p_sum = np.sum(p)

            if p_sum == 0:
                continue

            p /= p_sum
            cumulative_p = np.cumsum(p)
            j = np.searchsorted(cumulative_p, np.random.uniform())
            sb[j] = 1

        result[eligible_mask] = sb
        return result
