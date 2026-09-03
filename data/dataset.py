import torch
from typing import List, Tuple


class Solution:

    def batch_loader(
        self,
        raw_dataset: str,
        context_length: int,
        batch_size: int
    ) -> Tuple[List[List[str]], List[List[str]]]:

        # 1. Split raw text into tokens
        tokens = raw_dataset.split()

        # 2. Make random sampling reproducible
        torch.manual_seed(0)

        # Valid starting positions:
        # 0 ... len(tokens) - context_length - 1
        starts = torch.randint(
            0,
            len(tokens) - context_length,
            (batch_size,)
        )

        X = []
        Y = []

        # 3. Create input/target pairs
        for start in starts:
            start = start.item()

            X.append(
                tokens[start:start + context_length]
            )

            Y.append(
                tokens[start + 1:start + 1 + context_length]
            )

        return X, Y