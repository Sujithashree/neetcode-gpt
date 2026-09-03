import torch
import torch.nn as nn
import torch.nn.functional as F


class Solution:

    def train(
        self,
        model: nn.Module,
        data: torch.Tensor,
        epochs: int,
        context_length: int,
        batch_size: int,
        lr: float
    ) -> float:

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr
        )

        for epoch in range(epochs):

            # Reproducible sampling for this epoch
            torch.manual_seed(epoch)

            # Random starting positions
            starts = torch.randint(
                0,
                len(data) - context_length,
                (batch_size,)
            )

            # Build input and target batches
            X = torch.stack([
                data[start:start + context_length]
                for start in starts
            ])

            Y = torch.stack([
                data[start + 1:start + 1 + context_length]
                for start in starts
            ])

            # Forward pass
            logits = model(X)

            # logits: (B, T, vocab_size)
            # targets: (B, T)
            B, T, C = logits.shape

            # Flatten for cross entropy
            logits_flat = logits.reshape(B * T, C)
            targets_flat = Y.reshape(B * T)

            # Calculate loss
            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)