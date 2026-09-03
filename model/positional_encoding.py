import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_positional_encoding(
        self,
        seq_len: int,
        d_model: int
    ) -> NDArray[np.float64]:

        position = np.arange(seq_len)[:, np.newaxis]
        dimension = np.arange(0, d_model, 2)

        div_term = 10000 ** (dimension / d_model)

        pe = np.zeros((seq_len, d_model))

        pe[:, 0::2] = np.sin(position / div_term)
        pe[:, 1::2] = np.cos(position / div_term)

        return np.round(pe, 5)