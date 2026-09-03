import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)

        self.embedding = nn.Embedding(vocabulary_size, 16)
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # (B, T) -> (B, T, 16)
        embedded = self.embedding(x.long())

        # (B, T, 16) -> (B, 16)
        averaged = embedded.mean(dim=1)

        # (B, 16) -> (B, 1)
        output = self.linear(averaged)

        # Sigmoid -> values in [0, 1]
        output = self.sigmoid(output)

        return torch.round(output, decimals=4)