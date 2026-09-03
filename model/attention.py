import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()

        torch.manual_seed(0)

        self.key = nn.Linear(
            embedding_dim,
            attention_dim,
            bias=False
        )

        self.query = nn.Linear(
            embedding_dim,
            attention_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            attention_dim,
            bias=False
        )

        self.attention_dim = attention_dim

    def forward(
        self,
        embedded: TensorType[float]
    ) -> TensorType[float]:

        # Q, K, V
        K = self.key(embedded)
        Q = self.query(embedded)
        V = self.value(embedded)

        # Q @ K^T
        scores = Q @ K.transpose(-2, -1)

        # Scale by sqrt(d_k)
        scores = scores / (self.attention_dim ** 0.5)

        # Causal mask
        context_length = embedded.shape[1]

        mask = torch.tril(
            torch.ones(
                context_length,
                context_length,
                dtype=torch.bool,
                device=embedded.device
            )
        )

        # Future positions -> -infinity
        scores = scores.masked_fill(~mask, float("-inf"))

        # Convert scores to probabilities
        weights = torch.softmax(scores, dim=-1)

        # Weighted sum of V
        return weights @ V