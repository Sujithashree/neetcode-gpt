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

    def forward(
        self,
        embedded: TensorType[float]
    ) -> TensorType[float]:

        # X -> K, Q, V
        K = self.key(embedded)
        Q = self.query(embedded)
        V = self.value(embedded)

        # Q @ K^T / sqrt(d_k)
        scores = Q @ K.transpose(-2, -1)
        scores = scores / (K.size(-1) ** 0.5)

        # Causal mask: cannot look at future tokens
        context_length = embedded.size(1)

        mask = torch.triu(
            torch.ones(context_length, context_length),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(mask, float("-inf"))

        # Convert scores to attention weights
        attention = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        output = attention @ V

        return output