import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)

        std = math.sqrt(2.0 / (fan_in + fan_out))

        weights = torch.randn(fan_out, fan_in) * std

        return [
            [round(x.item(), 4) for x in row]
            for row in weights
        ]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)

        std = math.sqrt(2.0 / fan_in)

        weights = torch.randn(fan_out, fan_in) * std

        return [
            [round(x.item(), 4) for x in row]
            for row in weights
        ]

    def check_activations(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        init_type: str
    ) -> List[float]:

        torch.manual_seed(0)

        # Create all weights first
        weights = []

        for layer in range(num_layers):
            fan_in = input_dim if layer == 0 else hidden_dim
            fan_out = hidden_dim

            if init_type == "xavier":
                std = math.sqrt(2.0 / (fan_in + fan_out))

            elif init_type == "kaiming":
                std = math.sqrt(2.0 / fan_in)

            elif init_type == "random":
                std = 1.0

            else:
                raise ValueError("Invalid init_type")

            W = torch.randn(fan_out, fan_in) * std
            weights.append(W)

        # Generate input after generating weights
        x = torch.randn(input_dim)

        result = []

        for W in weights:
            x = x @ W.T
            x = torch.relu(x)

            result.append(round(x.std().item(), 2))

        return result