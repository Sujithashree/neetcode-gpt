from typing import Dict, List, Tuple


class Solution:

    def build_vocab(
        self,
        text: str
    ) -> Tuple[Dict[str, int], Dict[int, str]]:

        # Get unique characters and sort them
        chars = sorted(set(text))

        # Character -> integer
        stoi = {ch: i for i, ch in enumerate(chars)}

        # Integer -> character
        itos = {i: ch for i, ch in enumerate(chars)}

        return stoi, itos

    def encode(
        self,
        text: str,
        stoi: Dict[str, int]
    ) -> List[int]:

        return [stoi[ch] for ch in text]

    def decode(
        self,
        ids: List[int],
        itos: Dict[int, str]
    ) -> str:

        return "".join(itos[i] for i in ids)