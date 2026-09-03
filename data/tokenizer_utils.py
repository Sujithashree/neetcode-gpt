from typing import List, Dict


class Solution:

    def tokenize_numbers(
        self,
        numbers: List[int],
        vocab: Dict[str, int]
    ) -> List[List[str]]:

        result = []

        for number in numbers:
            text = str(number)
            tokens = []
            i = 0

            while i < len(text):
                best_token = None

                # Try every possible substring starting at i
                for j in range(i + 1, len(text) + 1):
                    candidate = text[i:j]

                    if candidate in vocab:
                        best_token = candidate

                if best_token is not None:
                    tokens.append(best_token)
                    i += len(best_token)
                else:
                    # No match: consume one character
                    tokens.append(text[i])
                    i += 1

            result.append(tokens)

        return result

    def count_tokens(
        self,
        text: str,
        vocab: Dict[str, int]
    ) -> int:

        count = 0
        i = 0

        while i < len(text):
            best_token = None

            for j in range(i + 1, len(text) + 1):
                candidate = text[i:j]

                if candidate in vocab:
                    best_token = candidate

            if best_token is not None:
                i += len(best_token)
            else:
                i += 1

            count += 1

        return count

    def fertility_score(
        self,
        text: str,
        vocab: Dict[str, int]
    ) -> float:

        token_count = self.count_tokens(text, vocab)

        # Split on spaces and ignore empty strings
        words = text.split()

        if not words:
            return 0.0

        return round(token_count / len(words), 4)