from typing import List
from collections import Counter


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # Start with individual characters
        tokens = list(corpus)

        merges = []

        for _ in range(num_merges):
            # If fewer than 2 tokens remain, no pair can be merged
            if len(tokens) < 2:
                break

            # Count adjacent pairs
            pair_counts = Counter(
                (tokens[i], tokens[i + 1])
                for i in range(len(tokens) - 1)
            )

            # Find the most frequent pair.
            # max() with (-count, pair) gives:
            #   1. highest count
            #   2. lexicographically smallest pair on ties
            best_pair = min(
                pair_counts,
                key=lambda pair: (-pair_counts[pair], pair)
            )

            a, b = best_pair

            # Record the merge
            merges.append([a, b])

            # Merge non-overlapping occurrences left to right
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