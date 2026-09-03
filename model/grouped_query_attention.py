import torch
import torch.nn as nn
from torchtyping import TensorType


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int
    ):
        super().__init__()

        torch.manual_seed(0)

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(
            model_dim,
            num_heads * self.head_dim,
            bias=False
        )

        self.k_proj = nn.Linear(
            model_dim,
            num_kv_heads * self.head_dim,
            bias=False
        )

        self.v_proj = nn.Linear(
            model_dim,
            num_kv_heads * self.head_dim,
            bias=False
        )

        self.output_proj = nn.Linear(
            num_heads * self.head_dim,
            model_dim,
            bias=False
        )

    def forward(
        self,
        x: TensorType[float]
    ) -> TensorType[float]:

        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape into heads
        # Q: (B, T, H*D) -> (B, H, T, D)
        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # K/V: (B, T, G*D) -> (B, G, T, D)
        k = k.view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        # Expand KV heads to match query heads
        heads_per_group = self.num_heads // self.num_kv_heads

        k = k.repeat_interleave(
            heads_per_group,
            dim=1
        )

        v = v.repeat_interleave(
            heads_per_group,
            dim=1
        )

        # Attention scores
        # (B, H, T, D) @ (B, H, D, T)
        # -> (B, H, T, T)
        scores = q @ k.transpose(-2, -1)

        scores = scores / (self.head_dim ** 0.5)

        # Causal mask
        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        # Attention weights
        weights = torch.softmax(scores, dim=-1)

        # Weighted values
        # (B, H, T, T) @ (B, H, T, D)
        # -> (B, H, T, D)
        out = weights @ v

        # Put heads back together
        # (B, H, T, D) -> (B, T, H, D)
        out = out.transpose(1, 2).contiguous()

        out = out.view(
            batch_size,
            seq_len,
            self.num_heads * self.head_dim
        )

        # Final projection
        return self.output_proj(out)