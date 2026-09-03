import numpy as np
from numpy.typing import NDArray

class Solution:
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z_stable = z - np.max(z)
        exp_z = np.exp(z_stable)
        probabilities = exp_z / np.sum(exp_z)

        return np.round(probabilities, 4)