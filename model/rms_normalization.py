import numpy as np
from typing import List

class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)

        # Calculate RMS
        rms = np.sqrt(np.mean(x ** 2) + eps)

        # Normalize and scale
        result = gamma * (x / rms)

        return np.round(result, 4).tolist()