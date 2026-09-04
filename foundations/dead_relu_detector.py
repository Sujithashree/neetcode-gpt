import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []
        hooks = []

        def make_hook():
            def hook(module, inputs, output):
                # output shape: (batch_size, number_of_neurons)

                # A neuron is dead if it outputs 0
                # for every sample in the batch
                dead = (output == 0).all(dim=0)

                dead_fraction = dead.float().mean().item()

                dead_fractions.append(round(dead_fraction, 4))

            return hook

        # Register a hook after every ReLU
        for layer in model.modules():
            if isinstance(layer, nn.ReLU):
                hooks.append(
                    layer.register_forward_hook(make_hook())
                )

        # Forward pass without gradients
        with torch.no_grad():
            model(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:

        if not dead_fractions:
            return "healthy"

        # 1. Severe dead neurons
        if any(x > 0.5 for x in dead_fractions):
            return "use_leaky_relu"

        # 2. First layer has too many dead neurons
        if dead_fractions[0] > 0.3:
            return "reinitialize"

        # 3. Dead neurons strictly increase with depth
        increasing = True

        for i in range(1, len(dead_fractions)):
            if dead_fractions[i] <= dead_fractions[i - 1]:
                increasing = False
                break

        if increasing and dead_fractions[-1] > 0.1:
            return "reduce_learning_rate"

        # 4. Healthy
        if max(dead_fractions) < 0.1:
            return "healthy"

        return "healthy"