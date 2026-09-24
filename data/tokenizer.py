from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # Start with individual characters
        tokens = list(corpus)

        merges = []

        for _ in range(num_merges):
            # Count adjacent pairs
            pair_count = {}

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_count[pair] = pair_count.get(pair, 0) + 1

            # No more pairs to merge
            if not pair_count:
                break

            # Maximum frequency, then lexicographically smallest pair
            best_pair = min(
                pair_count,
                key=lambda pair: (-pair_count[pair], pair)
            )

            a, b = best_pair

            # Record the merge
            merges.append([a, b])

            # Merge non-overlapping occurrences from left to right
            new_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == a
                    and tokens[i + 1] == b
                ):
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return merges