from abc import ABC, abstractmethod
import numpy as np


class SamplingMethod(ABC):
    @abstractmethod
    def sample(self, pik: np.ndarray) -> np.ndarray:
        pass
