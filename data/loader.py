import torch
from torchtyping import TensorType
from typing import Tuple


class Solution:

    def create_batches(
        self,
        data: TensorType[int],
        context_length: int,
        batch_size: int
    ) -> Tuple[TensorType[int], TensorType[int]]:

        torch.manual_seed(0)

        # Maximum valid starting position
        max_start = len(data) - context_length

        # Random starting positions
        starts = torch.randint(
            0,
            max_start,
            (batch_size,)
        )

        # Build X and Y
        X = torch.stack([
            data[start:start + context_length]
            for start in starts
        ])

        Y = torch.stack([
            data[start + 1:start + 1 + context_length]
            for start in starts
        ])

        return X, Y