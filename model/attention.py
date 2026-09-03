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

        # (B, T, D) -> (B, T, A)
        Q = self.query(embedded)
        K = self.key(embedded)
        V = self.value(embedded)

        # Attention scores: (B, T, T)
        scores = Q @ K.transpose(-2, -1)

        # Scale by sqrt(d_k)
        scores = scores / (self.query.out_features ** 0.5)

        # Causal mask: prevent attending to future tokens
        T = embedded.shape[1]
        mask = torch.triu(
            torch.ones(T, T, device=embedded.device),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(mask, float("-inf"))

        # Convert scores to attention probabilities
        attention_weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values: (B, T, A)
        output = attention_weights @ V

        return output