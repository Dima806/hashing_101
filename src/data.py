"""Synthetic data with known ground truth.

Everything is generated on the fly rather than downloaded, because the point of this project is
scoring approximations against the exact answer at a scale you control: a stream whose unique
count you *know*, frequencies you *know*, near-duplicate pairs you planted yourself.

Every generator takes an explicit seed. Nothing here is random in the sense of irreproducible.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_WORD_PARTS = [
    "data",
    "model",
    "hash",
    "bucket",
    "stream",
    "filter",
    "sketch",
    "memory",
    "error",
    "bound",
    "signal",
    "noise",
    "vector",
    "index",
    "token",
    "count",
    "unique",
    "probe",
    "collision",
    "bloom",
    "cardinality",
    "estimate",
    "corpus",
    "duplicate",
    "pipeline",
    "batch",
    "window",
    "shard",
    "cache",
    "lookup",
    "dictionary",
    "encoder",
    "feature",
    "bit",
    "array",
    "seed",
    "rate",
    "target",
]


@dataclass(frozen=True)
class Stream:
    """A stream of items whose exact statistics are known by construction."""

    items: list[str]
    universe: list[str]
    true_counts: Counter[str] = field(repr=False)

    @property
    def n_items(self) -> int:
        """Length of the stream, duplicates included."""
        return len(self.items)

    @property
    def true_cardinality(self) -> int:
        """The exact number of distinct items - what HyperLogLog has to guess."""
        return len(self.true_counts)

    def top_k(self, k: int = 10) -> list[tuple[str, int]]:
        """The k heaviest hitters, exactly - what Count-Min has to guess."""
        return self.true_counts.most_common(k)

    def absent_items(self, n: int, prefix: str = "absent-") -> list[str]:
        """Items guaranteed not to be in the stream, for measuring false positives."""
        return [f"{prefix}{i:09d}" for i in range(n)]


def generate_stream(
    n_items: int = 200_000,
    n_unique: int = 20_000,
    seed: int = 0,
    distribution: str = "zipf",
    zipf_exponent: float = 1.2,
    prefix: str = "item-",
) -> Stream:
    """Generate a stream with exactly ``n_unique`` distinct items.

    ``distribution="zipf"`` gives the heavy-tailed frequencies real traffic has (a few items
    dominate - which is what makes heavy-hitter detection interesting); ``"uniform"`` spreads
    counts evenly. Every distinct item is guaranteed to appear at least once, so the cardinality
    is exactly ``n_unique`` rather than approximately it.
    """
    if n_items < n_unique:
        raise ValueError("n_items must be at least n_unique for the cardinality to be exact")
    rng = np.random.default_rng(seed)
    universe = [f"{prefix}{i:08d}" for i in range(n_unique)]

    if distribution == "uniform":
        draws = rng.integers(0, n_unique, size=n_items - n_unique)
    elif distribution == "zipf":
        weights = 1.0 / np.power(np.arange(1, n_unique + 1), zipf_exponent)
        weights /= weights.sum()
        draws = rng.choice(n_unique, size=n_items - n_unique, p=weights)
    else:
        raise ValueError("distribution must be 'zipf' or 'uniform'")

    indices = np.concatenate([np.arange(n_unique), draws])
    rng.shuffle(indices)
    items = [universe[int(i)] for i in indices]
    return Stream(items=items, universe=universe, true_counts=Counter(items))


def generate_unique_items(n: int, seed: int = 0, prefix: str = "user-") -> list[str]:
    """``n`` distinct identifiers in shuffled order - the input for cardinality experiments."""
    rng = np.random.default_rng(seed)
    items = [f"{prefix}{i:09d}" for i in range(n)]
    rng.shuffle(items)
    return items


@dataclass(frozen=True)
class Corpus:
    """A document set with planted near-duplicates, so recall can actually be measured."""

    documents: list[str]
    ids: list[str]
    duplicate_pairs: set[tuple[int, int]] = field(repr=False)

    @property
    def n_documents(self) -> int:
        """How many documents the corpus holds."""
        return len(self.documents)

    @property
    def n_all_pairs(self) -> int:
        """The quadratic baseline: every pair an exhaustive search would have to compare."""
        return self.n_documents * (self.n_documents - 1) // 2


def _random_document(
    rng: np.random.Generator, n_words: int, vocabulary: Sequence[str]
) -> list[str]:
    return [vocabulary[int(i)] for i in rng.integers(0, len(vocabulary), size=n_words)]


def generate_text_corpus(
    n_docs: int = 400,
    n_near_duplicates: int = 40,
    doc_words: int = 60,
    vocabulary_size: int = 400,
    edit_fraction: float = 0.04,
    seed: int = 0,
) -> Corpus:
    """Generate documents, then plant near-duplicates by lightly editing some of them.

    ``duplicate_pairs`` holds the planted (original, copy) index pairs - the ground truth LSH
    recall is measured against.
    """
    if n_near_duplicates > n_docs:
        raise ValueError("n_near_duplicates cannot exceed n_docs")
    rng = np.random.default_rng(seed)
    vocabulary = [
        f"{word}{suffix}"
        for suffix in range(1 + vocabulary_size // len(_WORD_PARTS))
        for word in _WORD_PARTS
    ][:vocabulary_size]

    n_originals = n_docs - n_near_duplicates
    documents = [_random_document(rng, doc_words, vocabulary) for _ in range(n_originals)]
    duplicate_pairs: set[tuple[int, int]] = set()

    sources = rng.choice(n_originals, size=n_near_duplicates, replace=False)
    for source in sources:
        copy = list(documents[int(source)])
        n_edits = max(1, int(round(edit_fraction * doc_words)))
        positions = rng.choice(doc_words, size=n_edits, replace=False)
        for position in positions:
            copy[int(position)] = vocabulary[int(rng.integers(0, len(vocabulary)))]
        documents.append(copy)
        duplicate_pairs.add((int(source), len(documents) - 1))

    return Corpus(
        documents=[" ".join(words) for words in documents],
        ids=[f"doc-{i:05d}" for i in range(len(documents))],
        duplicate_pairs=duplicate_pairs,
    )


def generate_categorical_table(
    n_rows: int = 20_000,
    n_categories: int = 5_000,
    n_informative: int = 200,
    noise: float = 0.3,
    seed: int = 0,
) -> pd.DataFrame:
    """A table with one very high-cardinality column and a target that depends on it.

    ``user_id`` has ``n_categories`` distinct values of which only ``n_informative`` carry signal,
    which is exactly the situation feature hashing is used for: a column too wide to one-hot, most
    of it noise. The target is linear in the informative categories plus Gaussian noise, so a
    ridge model on hashed features has something real to find - and its R^2 falls in a measurable
    way as buckets shrink and collisions blur the signal.
    """
    rng = np.random.default_rng(seed)
    cities = ["berlin", "lisbon", "warsaw", "kyiv", "porto", "tallinn", "vienna", "riga"]
    devices = ["ios", "android", "web", "tv"]

    user_index = rng.integers(0, n_categories, size=n_rows)
    city_index = rng.integers(0, len(cities), size=n_rows)
    device_index = rng.integers(0, len(devices), size=n_rows)

    user_effect = np.zeros(n_categories)
    informative = rng.choice(n_categories, size=min(n_informative, n_categories), replace=False)
    user_effect[informative] = rng.normal(0.0, 1.0, size=len(informative))
    city_effect = rng.normal(0.0, 0.5, size=len(cities))
    device_effect = rng.normal(0.0, 0.3, size=len(devices))

    target = (
        user_effect[user_index]
        + city_effect[city_index]
        + device_effect[device_index]
        + rng.normal(0.0, noise, size=n_rows)
    )
    return pd.DataFrame(
        {
            "user_id": [f"user-{int(i):06d}" for i in user_index],
            "city": [cities[int(i)] for i in city_index],
            "device": [devices[int(i)] for i in device_index],
            "y": target,
        }
    )
