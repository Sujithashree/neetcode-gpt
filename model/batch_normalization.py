import numpy as np
from typing import Tuple, List

class Solution:
    def batch_norm(
        self,
        x: List[List[float]],
        gamma: List[float],
        beta: List[float],
        running_mean: List[float],
        running_var: List[float],
        momentum: float,
        eps: float,
        training: bool
    ) -> Tuple[List[List[float]], List[float], List[float]]:

        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)
        beta = np.array(beta, dtype=float)
        running_mean = np.array(running_mean, dtype=float)
        running_var = np.array(running_var, dtype=float)

        if training:
            # Mean and variance across the batch
            batch_mean = np.mean(x, axis=0)
            batch_var = np.mean((x - batch_mean) ** 2, axis=0)

            # Normalize using batch statistics
            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)

            # Update running statistics
            running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            running_var = (1 - momentum) * running_var + momentum * batch_var

        else:
            # Normalize using running statistics
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)

        # Scale and shift
        y = gamma * x_hat + beta

        return (
            np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )