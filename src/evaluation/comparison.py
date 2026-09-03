"""Exact versus approximate: memory, speed, accuracy, measured on the same data.

Every row of these tables is a real run. The exact structure is a Python ``set`` or ``Counter`` or
an all-pairs loop - the thing a practitioner would reach for first - and the approximate structure
is the one built in this project. The interesting column is always the last one: what the error
bought.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Sequence

import pandas as pd

from src.data import Corpus, generate_unique_items
from src.probabilistic.bloom import BloomFilter
from src.probabilistic.hyperloglog import HyperLogLog
from src.similarity.lsh import MinHashLSH
from src.similarity.minhash import MinHasher, jaccard, shingles


def deep_sizeof(items: Iterable[object]) -> int:
    """Bytes a Python container really costs: the container plus every object inside it."""
    collected = list(items)
    container: set[object] = set(collected)
    return sys.getsizeof(container) + sum(sys.getsizeof(item) for item in container)


def compare_membership(
    n_items: int = 100_000,
    target_fp_rate: float = 0.01,
    n_queries: int = 50_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Exact ``set`` versus Bloom filter on the same membership workload.

    The Bloom row is wrong sometimes - on purpose, at the rate that was asked for - and never
    wrong in the direction that matters for deduplication.
    """
    items = generate_unique_items(n_items, seed=seed)
    absent = [f"absent-{i:09d}" for i in range(n_queries)]

    start = time.perf_counter()
    exact = set(items)
    exact_build = time.perf_counter() - start

    start = time.perf_counter()
    exact_hits = sum(1 for item in absent if item in exact)
    exact_query = time.perf_counter() - start

    start = time.perf_counter()
    bloom = BloomFilter(n_items, target_fp_rate, seed=seed)
    bloom.add_many(items)
    bloom_build = time.perf_counter() - start

    start = time.perf_counter()
    bloom_hits = int(bloom.contains_many(absent).sum())
    bloom_query = time.perf_counter() - start

    false_negatives = int((~bloom.contains_many(items)).sum())

    return pd.DataFrame(
        [
            {
                "structure": "exact set",
                "memory_bytes": deep_sizeof(items),
                "build_seconds": exact_build,
                "query_seconds": exact_query,
                "false_positive_rate": exact_hits / n_queries,
                "false_negatives": 0,
            },
            {
                "structure": f"bloom filter (p={target_fp_rate})",
                "memory_bytes": bloom.memory_bytes(),
                "build_seconds": bloom_build,
                "query_seconds": bloom_query,
                "false_positive_rate": bloom_hits / n_queries,
                "false_negatives": false_negatives,
            },
        ]
    )


def compare_cardinality(
    n_unique: int = 200_000,
    precision: int = 11,
    seed: int = 0,
) -> pd.DataFrame:
    """Exact ``set`` versus HyperLogLog on counting distinct items.

    The headline row: kilobytes against megabytes, for an answer that is wrong by about 2%.
    """
    items = generate_unique_items(n_unique, seed=seed)

    start = time.perf_counter()
    exact = set(items)
    exact_seconds = time.perf_counter() - start

    start = time.perf_counter()
    sketch = HyperLogLog(precision=precision, seed=seed)
    sketch.add_many(items)
    estimate = sketch.estimate()
    sketch_seconds = time.perf_counter() - start

    return pd.DataFrame(
        [
            {
                "structure": "exact set",
                "memory_bytes": deep_sizeof(items),
                "seconds": exact_seconds,
                "estimate": float(len(exact)),
                "relative_error": 0.0,
            },
            {
                "structure": f"hyperloglog (p={precision})",
                "memory_bytes": sketch.packed_memory_bytes(),
                "seconds": sketch_seconds,
                "estimate": estimate,
                "relative_error": (estimate - n_unique) / n_unique,
            },
        ]
    )


def compare_near_duplicate_search(
    corpus: Corpus,
    num_bands: int = 32,
    rows_per_band: int = 4,
    shingle_size: int = 5,
    similarity_threshold: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """All-pairs comparison versus MinHash + LSH on the same corpus.

    Both rows find near-duplicates; only one of them has to look at every pair.
    """
    shingle_sets = [shingles(document, shingle_size) for document in corpus.documents]
    n_docs = corpus.n_documents

    start = time.perf_counter()
    truth = {
        (i, j)
        for i in range(n_docs)
        for j in range(i + 1, n_docs)
        if jaccard(shingle_sets[i], shingle_sets[j]) >= similarity_threshold
    }
    exhaustive_seconds = time.perf_counter() - start

    start = time.perf_counter()
    hasher = MinHasher(num_perm=num_bands * rows_per_band, seed=seed)
    index = MinHashLSH(num_bands=num_bands, rows_per_band=rows_per_band, seed=seed)
    for i, shingle_set in enumerate(shingle_sets):
        index.insert(i, hasher.signature(shingle_set))
    candidates = {tuple(sorted(pair)) for pair in index.candidate_pairs()}
    verified = {
        pair
        for pair in candidates
        if jaccard(shingle_sets[pair[0]], shingle_sets[pair[1]]) >= similarity_threshold
    }
    lsh_seconds = time.perf_counter() - start

    return pd.DataFrame(
        [
            {
                "method": "all pairs",
                "comparisons": corpus.n_all_pairs,
                "pairs_found": len(truth),
                "recall": 1.0,
                "seconds": exhaustive_seconds,
            },
            {
                "method": f"minhash + lsh (b={num_bands}, r={rows_per_band})",
                "comparisons": len(candidates),
                "pairs_found": len(verified),
                "recall": len(verified & truth) / len(truth) if truth else float("nan"),
                "seconds": lsh_seconds,
            },
        ]
    )


def decision_guide() -> pd.DataFrame:
    """The closing table of notebook 06: which structure, for which question.

    The rule underneath every row is the same - if the exact answer fits in memory, take it; the
    approximate structures exist for the case where it does not, and each one gives up exactly one
    thing in exchange.
    """
    return pd.DataFrame(
        [
            {
                "question": "Is this item in my set? (and it fits in memory)",
                "structure": "set / dict",
                "memory": "O(n), ~100 bytes per item",
                "error": "none",
                "gives up": "nothing, until it does not fit",
            },
            {
                "question": "Have I seen this item before, at scale?",
                "structure": "Bloom filter",
                "memory": "~9.6 bits per item at 1%",
                "error": "false positives only, tunable",
                "gives up": "certainty on 'yes'; never on 'no'",
            },
            {
                "question": "Have I seen it, and can I forget items?",
                "structure": "Counting Bloom filter",
                "memory": "8x a Bloom filter",
                "error": "false positives only",
                "gives up": "memory, in exchange for deletion",
            },
            {
                "question": "How many distinct items are there?",
                "structure": "HyperLogLog",
                "memory": "1.5 KB at 2.3% error",
                "error": "~1.04/sqrt(m) relative",
                "gives up": "the items themselves; only the count survives",
            },
            {
                "question": "How often did each item appear?",
                "structure": "Count-Min Sketch",
                "memory": "d x w counters, fixed",
                "error": "overestimates by <= eps*N",
                "gives up": "accuracy on rare items; heavy hitters stay reliable",
            },
            {
                "question": "Which documents are near-duplicates?",
                "structure": "MinHash + LSH",
                "memory": "128 numbers per document",
                "error": "misses some pairs near the threshold",
                "gives up": "exhaustive comparison, and the quadratic cost with it",
            },
            {
                "question": "How do I encode a million categories?",
                "structure": "Feature hashing",
                "memory": "n_buckets per row, fixed",
                "error": "collisions blur categories",
                "gives up": "interpretability of a single column",
            },
        ]
    )


def memory_summary(structures: Sequence[tuple[str, int, int]]) -> pd.DataFrame:
    """Tabulate ``(name, exact_bytes, approximate_bytes)`` triples into a savings table."""
    return pd.DataFrame(
        [
            {
                "structure": name,
                "exact_bytes": exact_bytes,
                "approximate_bytes": approximate_bytes,
                "savings_factor": exact_bytes / approximate_bytes if approximate_bytes else None,
            }
            for name, exact_bytes, approximate_bytes in structures
        ]
    )
