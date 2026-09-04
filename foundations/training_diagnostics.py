import torch
import torch.nn as nn


class Solution:

    def compute_activation_stats(self, model, x):
        stats = []
        hooks = []

        def hook_fn(module, inputs, output):
            mean = output.mean().item()
            std = output.std().item()

            # A neuron is dead if output <= 0
            # for every sample in the batch
            dead = (output <= 0).all(dim=0)
            dead_fraction = dead.float().mean().item()

            stats.append({
                "mean": round(mean, 4),
                "std": round(std, 4),
                "dead_fraction": round(dead_fraction, 4)
            })

        # Hook every Linear layer
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                hooks.append(layer.register_forward_hook(hook_fn))

        with torch.no_grad():
            model(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return stats

    def compute_gradient_stats(self, model, x, y):
        model.zero_grad()

        # Forward
        output = model(x)

        # MSE loss
        loss = nn.MSELoss()(output, y)

        # Backward
        loss.backward()

        stats = []

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad

                stats.append({
                    "mean": round(grad.mean().item(), 4),
                    "std": round(grad.std().item(), 4),
                    "norm": round(torch.norm(grad).item(), 4)
                })

        return stats

    def diagnose(self, activation_stats, gradient_stats):

        # 1. Dead neurons
        for stat in activation_stats:
            if stat["dead_fraction"] > 0.5:
                return "dead_neurons"

        # 2. Exploding gradients
        for stat in gradient_stats:
            if stat["norm"] > 1000:
                return "exploding_gradients"

        # 3. Vanishing gradients in last layer
        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # 4. Check activation standard deviation
        for stat in activation_stats:
            if stat["std"] < 0.1:
                return "vanishing_gradients"

            if stat["std"] > 10.0:
                return "exploding_gradients"

        return "healthy"