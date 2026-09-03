import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution:
    def generate(
        self,
        model,
        new_chars: int,
        context: TensorType[int],
        context_length: int,
        int_to_char: dict
    ) -> str:

        # Fixed random generator for reproducible output
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()

        generated = []

        for _ in range(new_chars):

            # Keep only the most recent context_length tokens
            context = context[:, -context_length:]

            # Model output:
            # (1, seq_len, vocab_size)
            logits = model(context)

            # Get logits for the last token position
            # (1, vocab_size)
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Sample one token
            # Shape: (1, 1)
            next_token = torch.multinomial(
                probs,
                num_samples=1,
                generator=generator
            )

            # Add token to context
            context = torch.cat(
                [context, next_token],
                dim=1
            )

            # Convert token ID to character
            token_id = next_token.item()
            generated.append(int_to_char[token_id])

        return "".join(generated)