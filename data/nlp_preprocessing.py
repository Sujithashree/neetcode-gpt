import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(
        self,
        positive: List[str],
        negative: List[str]
    ) -> TensorType[float]:

        # Combine all sentences
        sentences = positive + negative

        # Build sorted vocabulary
        vocab = sorted(set(
            word
            for sentence in sentences
            for word in sentence.split()
        ))

        # Word -> ID, starting from 1
        word_to_id = {
            word: i + 1
            for i, word in enumerate(vocab)
        }

        # Convert sentences to lists of IDs
        encoded = []

        for sentence in sentences:
            ids = [word_to_id[word] for word in sentence.split()]
            encoded.append(torch.tensor(ids, dtype=torch.float))

        # Pad shorter sentences with 0
        result = torch.nn.utils.rnn.pad_sequence(
            encoded,
            batch_first=True,
            padding_value=0
        )

        return result