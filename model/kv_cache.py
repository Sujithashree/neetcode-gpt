import torch
import torch.nn as nn
from typing import Tuple, Optional


class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # First call: initialize cache
        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v

        # Later calls: append along sequence dimension
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)

        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None


class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()

        torch.manual_seed(0)

        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None
    ) -> Tuple[torch.Tensor, KVCache]:

        # Create cache if this is the first call
        if kv_cache is None:
            kv_cache = KVCache()

        # Project only the new tokens
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Number of tokens that existed before adding new K/V
        past_len = (
            0 if kv_cache.cache_k is None
            else kv_cache.cache_k.shape[1]
        )

        # Add new K/V to cache
        full_k, full_v = kv_cache.update(k, v)

        # Scaled dot-product attention
        scores = q @ full_k.transpose(1, 2)

        scores = scores / (full_k.shape[-1] ** 0.5)

        # Causal mask
        #
        # New tokens have positions:
        # past_len, past_len + 1, ...
        #
        # They can attend to all cached tokens up to their own position.
        seq_len = x.shape[1]
        total_len = full_k.shape[1]

        query_positions = torch.arange(
            past_len,
            past_len + seq_len,
            device=x.device
        ).unsqueeze(1)

        key_positions = torch.arange(
            total_len,
            device=x.device
        ).unsqueeze(0)

        mask = key_positions > query_positions

        scores = scores.masked_fill(mask, float("-inf"))

        # Softmax over keys
        attention = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        output = attention @ full_v

        # Round to 4 decimal places
        output = torch.round(output * 10000) / 10000

        return output, kv_cache