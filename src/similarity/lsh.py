"""Locality-sensitive hashing: find the near-duplicates without comparing all the pairs.

Comparing every document with every other is quadratic: 100k documents is 5 billion comparisons,
which is why teams decide near-duplicate detection is a hard problem. It is not.

Cut each MinHash signature into ``b`` bands of ``r`` rows and hash each band into a bucket. Two
documents become *candidates* if they collide in at least one band. Because a band matches with
probability ``s^r`` when the true similarity is ``s``::

    P(candidate | similarity s) = 1 - (1 - s^r)^b

That is an S-curve with a knee near ``(1/b)^(1/r)``: below the knee almost nothing is a candidate,
above it almost everything is. Tune the knee to the similarity you care about and near-duplicate
detection becomes a dictionary lookup - near-linear instead of quadratic (Indyk and Motwani, 1998).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from src.core.hashes import hash64

Key = Any
"""A document key. Anything hashable works - callers usually use an index or a document id,
and typing it loosely lets them sort or compare their own key type."""


def candidate_probability(similarity: float, num_bands: int, rows_per_band: int) -> float:
    """``1 - (1 - s^r)^b`` - the S-curve that says which pairs LSH will even look at."""
    return 1.0 - (1.0 - similarity**rows_per_band) ** num_bands


def approximate_threshold(num_bands: int, rows_per_band: int) -> float:
    """``(1/b)^(1/r)`` - where the S-curve turns over, the de facto similarity cutoff."""
    return float((1.0 / num_bands) ** (1.0 / rows_per_band))


def choose_parameters(num_perm: int, target_threshold: float) -> tuple[int, int]:
    """Pick ``(bands, rows)`` with ``bands * rows == num_perm`` closest to a target threshold."""
    if not 0.0 < target_threshold < 1.0:
        raise ValueError("target_threshold must be in (0, 1)")
    best: tuple[int, int] = (1, num_perm)
    best_gap = float("inf")
    for bands in range(1, num_perm + 1):
        if num_perm % bands:
            continue
        rows = num_perm // bands
        gap = abs(approximate_threshold(bands, rows) - target_threshold)
        if gap < best_gap:
            best, best_gap = (bands, rows), gap
    return best


@dataclass(frozen=True)
class LSHStats:
    """How much work the index saved, in the only unit that matters: pairs compared."""

    n_keys: int
    n_buckets: int
    n_candidate_pairs: int
    n_all_pairs: int

    @property
    def work_ratio(self) -> float:
        """Candidate pairs as a fraction of the all-pairs baseline. Smaller is the whole point."""
        return self.n_candidate_pairs / self.n_all_pairs if self.n_all_pairs else 0.0


class MinHashLSH:
    """A banded index over MinHash signatures: insert documents, query for near-duplicates.

    >>> index = MinHashLSH(num_bands=16, rows_per_band=8)
    >>> index.num_perm
    128
    """

    def __init__(self, num_bands: int = 16, rows_per_band: int = 8, seed: int = 0) -> None:
        if num_bands < 1 or rows_per_band < 1:
            raise ValueError("num_bands and rows_per_band must be positive")
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.num_perm = num_bands * rows_per_band
        self.seed = seed
        self._buckets: dict[tuple[int, int], list[Key]] = defaultdict(list)
        self._signatures: dict[Key, np.ndarray] = {}

    def __repr__(self) -> str:
        return (
            f"MinHashLSH(num_bands={self.num_bands}, rows_per_band={self.rows_per_band}, "
            f"threshold~{self.threshold():.2f})"
        )

    def __len__(self) -> int:
        return len(self._signatures)

    def _band_keys(self, signature: np.ndarray) -> list[tuple[int, int]]:
        if signature.shape[0] != self.num_perm:
            raise ValueError(
                f"signature has {signature.shape[0]} values, expected {self.num_perm}"
            )
        keys = []
        for band in range(self.num_bands):
            start = band * self.rows_per_band
            chunk = signature[start : start + self.rows_per_band]
            keys.append((band, hash64(chunk.tobytes(), self.seed + band)))
        return keys

    def insert(self, key: Key, signature: np.ndarray) -> None:
        """Index one document under each of its band buckets."""
        for bucket in self._band_keys(signature):
            self._buckets[bucket].append(key)
        self._signatures[key] = signature

    def insert_many(self, keys: Sequence[Key], signatures: np.ndarray) -> None:
        """Index a batch: ``signatures`` is ``(n_keys, num_perm)``."""
        for key, signature in zip(keys, signatures, strict=True):
            self.insert(key, signature)

    def query(self, signature: np.ndarray, exclude: Key = None) -> set[Key]:
        """Keys sharing at least one band bucket with this signature - the candidate set."""
        found: set[Key] = set()
        for bucket in self._band_keys(signature):
            found.update(self._buckets.get(bucket, ()))
        found.discard(exclude)
        return found

    def candidate_pairs(self) -> set[tuple[Key, Key]]:
        """Every pair that collides in at least one band, deduplicated."""
        pairs: set[tuple[Key, Key]] = set()
        for members in self._buckets.values():
            if len(members) < 2:
                continue
            unique = sorted(set(members), key=repr)
            pairs.update(combinations(unique, 2))
        return pairs

    def threshold(self) -> float:
        """The similarity where this banding turns on: ``(1/b)^(1/r)``."""
        return approximate_threshold(self.num_bands, self.rows_per_band)

    def probability(self, similarity: float) -> float:
        """Probability a pair at this true similarity becomes a candidate."""
        return candidate_probability(similarity, self.num_bands, self.rows_per_band)

    def stats(self) -> LSHStats:
        """Index size and the work it saves against the all-pairs baseline."""
        n_keys = len(self._signatures)
        return LSHStats(
            n_keys=n_keys,
            n_buckets=len(self._buckets),
            n_candidate_pairs=len(self.candidate_pairs()),
            n_all_pairs=n_keys * (n_keys - 1) // 2,
        )

    def memory_bytes(self) -> int:
        """Approximate index size: the signatures plus the bucket lists."""
        signature_bytes = sum(int(sig.nbytes) for sig in self._signatures.values())
        bucket_bytes = sum(8 * len(members) for members in self._buckets.values())
        return signature_bytes + bucket_bytes
