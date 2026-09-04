import numpy as np
from numpy.typing import NDArray

class Solution:
    def lookup(
        self,
        embeddings: NDArray[np.float64],
        token_ids: NDArray[np.int64]
    ) -> NDArray[np.float64]:

        result = embeddings[token_ids]

        return np.round(result, 5)