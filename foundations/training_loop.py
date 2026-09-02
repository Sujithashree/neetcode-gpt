import numpy as np
from numpy.typing import NDArray
from typing import Tuple

class Solution:
    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        epochs: int,
        lr: float
    ) -> Tuple[NDArray[np.float64], float]:

        n = X.shape[0]

        # Initialize weights and bias
        w = np.zeros(X.shape[1])
        b = 0.0

        for _ in range(epochs):
            # Prediction
            y_hat = X @ w + b

            # Error
            error = y_hat - y

            # Gradients
            dw = (2 / n) * (X.T @ error)
            db = (2 / n) * np.sum(error)

            # Update weights and bias
            w = w - lr * dw
            b = b - lr * db

        return np.round(w, 5), round(b, 5)