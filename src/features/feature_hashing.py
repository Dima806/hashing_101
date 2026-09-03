"""Feature hashing: the same trick, pointed at a machine learning problem.

You have a categorical column with two million distinct values (URLs, user IDs, product SKUs).
One-hot encoding needs a two-million-wide vector and a vocabulary you must build, store, and keep
in sync between training and serving. Feature hashing skips all of it: hash the category, take the
remainder modulo ``n_buckets``, and that is the column index. No vocabulary, fixed width, and
unseen categories at serving time simply land somewhere instead of breaking (Weinberger et al.,
2009).

The cost is collisions: two categories can share a bucket, and the model cannot tell them apart.
The **signed** variant (``alternate_sign=True``) uses one spare hash bit to add each category with
a random +1 or -1, so colliding contributions cancel in expectation rather than accumulating -
the estimator stays unbiased even though individual features are contaminated.

Picking ``n_buckets`` is the whole design decision, and it is a real tradeoff rather than a
mystery: too few and collisions destroy signal, too many and the vector is wastefully sparse.
Notebook 06 plots model quality against it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from src.core.hashes import hash64

Row = Mapping[str, object] | Sequence[str]


def expected_collision_rate(n_tokens: int, n_buckets: int) -> float:
    """Expected fraction of categories that share a bucket with another category.

    With ``V`` categories in ``B`` buckets, the expected number of occupied buckets is
    ``B (1 - (1 - 1/B)^V)``, so this is ``1 - occupied / V`` - the balls-in-bins answer the
    measured rate in :meth:`FeatureHasher.collision_rate` should track.
    """
    if n_tokens <= 0:
        return 0.0
    occupied = n_buckets * (1.0 - (1.0 - 1.0 / n_buckets) ** n_tokens)
    return max(0.0, 1.0 - occupied / n_tokens)


class FeatureHasher:
    """Map categorical values into a fixed-width numeric vector, with the collisions visible.

    >>> hasher = FeatureHasher(n_buckets=8, seed=0)
    >>> matrix = hasher.transform([{"city": "berlin", "device": "ios"}])
    >>> matrix.shape
    (1, 8)
    """

    def __init__(self, n_buckets: int = 256, seed: int = 0, alternate_sign: bool = True) -> None:
        if n_buckets < 1:
            raise ValueError("n_buckets must be positive")
        self.n_buckets = n_buckets
        self.seed = seed
        self.alternate_sign = alternate_sign

    def __repr__(self) -> str:
        return (
            f"FeatureHasher(n_buckets={self.n_buckets}, seed={self.seed}, "
            f"alternate_sign={self.alternate_sign})"
        )

    def bucket_of(self, token: str) -> int:
        """The column this category lands in."""
        return hash64(token, self.seed) % self.n_buckets

    def sign_of(self, token: str) -> int:
        """+1 or -1 from a spare hash bit, so colliding categories cancel instead of piling up."""
        if not self.alternate_sign:
            return 1
        return 1 if (hash64(token, self.seed) >> 63) & 1 == 0 else -1

    @staticmethod
    def _tokens_and_weights(row: Row) -> list[tuple[str, float]]:
        """Normalise a row into ``(token, weight)`` pairs.

        A mapping becomes ``"feature=value"`` tokens (so ``city=berlin`` and ``device=berlin``
        stay distinct); a numeric mapping value becomes the weight; a plain sequence of strings
        becomes unit-weight tokens.
        """
        if isinstance(row, Mapping):
            pairs = []
            for feature, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    pairs.append((str(feature), float(value)))
                else:
                    pairs.append((f"{feature}={value}", 1.0))
            return pairs
        return [(str(token), 1.0) for token in row]

    def transform_row(self, row: Row) -> np.ndarray:
        """Hash one row into a dense vector of length ``n_buckets``."""
        vector = np.zeros(self.n_buckets, dtype=np.float64)
        for token, weight in self._tokens_and_weights(row):
            vector[self.bucket_of(token)] += self.sign_of(token) * weight
        return vector

    def transform(self, rows: Sequence[Row]) -> np.ndarray:
        """Hash many rows into a dense ``(n_rows, n_buckets)`` matrix."""
        matrix = np.zeros((len(rows), self.n_buckets), dtype=np.float64)
        for i, row in enumerate(rows):
            matrix[i] = self.transform_row(row)
        return matrix

    def bucket_assignments(self, tokens: Iterable[str]) -> dict[str, int]:
        """Which bucket each category was sent to - the raw material for collision counting."""
        return {token: self.bucket_of(token) for token in tokens}

    def collision_rate(self, tokens: Iterable[str]) -> float:
        """Measured fraction of categories sharing a bucket with at least one other."""
        assignments = self.bucket_assignments(tokens)
        if not assignments:
            return 0.0
        distinct_buckets = len(set(assignments.values()))
        return 1.0 - distinct_buckets / len(assignments)

    def colliding_groups(self, tokens: Iterable[str]) -> dict[int, list[str]]:
        """Buckets holding more than one category, and who is in them.

        Notebook 06 prints a few of these: seeing ``"city=lisbon"`` and ``"device=android"`` share
        a column is what turns "collisions" from a word into a thing that happened.
        """
        groups: dict[int, list[str]] = {}
        for token, bucket in self.bucket_assignments(tokens).items():
            groups.setdefault(bucket, []).append(token)
        return {bucket: sorted(members) for bucket, members in groups.items() if len(members) > 1}

    def memory_bytes(self, n_rows: int) -> int:
        """Bytes a dense hashed matrix occupies, against ``n_categories * n_rows`` for one-hot."""
        return int(n_rows * self.n_buckets * np.dtype(np.float64).itemsize)
