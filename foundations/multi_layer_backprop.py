import numpy as np
from typing import List

class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float]
    ) -> dict:

        # Convert to NumPy arrays
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # ---------- Forward pass ----------
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)

        z2 = W2 @ a1 + b2
        y_pred = z2

        # MSE loss
        loss = np.mean((y_pred - y_true) ** 2)

        # ---------- Backward pass ----------
        # dL/dz2
        dz2 = 2 * (y_pred - y_true) / y_true.size

        # Gradients for second layer
        dW2 = np.outer(dz2, a1)
        db2 = dz2

        # Gradient flowing into hidden layer
        da1 = W2.T @ dz2

        # ReLU derivative
        dz1 = da1 * (z1 > 0)

        # Gradients for first layer
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist()
        }