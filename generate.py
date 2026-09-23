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

        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr
        )

        loss = None

        for epoch in range(epochs):

            # Reproducible batch sampling
            torch.manual_seed(epoch)

            # Random starting positions
            starts = torch.randint(
                0,
                len(data) - context_length,
                (batch_size,)
            )

            # Build input and target batches
            X = torch.stack([
                data[i:i + context_length]
                for i in starts
            ])

            Y = torch.stack([
                data[i + 1:i + context_length + 1]
                for i in starts
            ])

            # Forward pass
            logits = model(X)

            # logits: (B, T, C)
            # targets: (B, T)
            #
            # Cross entropy expects:
            # input  -> (N, C)
            # target -> (N)

            B, T, C = logits.shape

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

        # Return final loss rounded to 4 decimals
        return round(loss.item(), 4)