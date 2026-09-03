"""Measured error versus memory, for every structure in the project.

Each function here runs the real structure over generated data with known ground truth and returns
a tidy DataFrame: what the theory predicted, what actually happened, and how many bytes it took.
The notebooks plot these; the tests assert against the theoretical bands.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.core.hashes import avalanche_matrix, chi_square_uniformity, clumping_hash, hash64
from src.data import Corpus, Stream, generate_stream, generate_unique_items
from src.features.feature_hashing import FeatureHasher, expected_collision_rate
from src.probabilistic.bloom import BloomFilter, sizing
from src.probabilistic.count_min import CountMinSketch
from src.probabilistic.hyperloglog import HyperLogLog, standard_error
from src.similarity.lsh import MinHashLSH, approximate_threshold
from src.similarity.minhash import MinHasher, jaccard, shingles


def hash_quality_report(
    values: Sequence[object] | None = None,
    n_buckets: int = 256,
    n_avalanche_samples: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Uniformity and avalanche for a good hash and a deliberately bad one, side by side."""
    if values is None:
        values = [f"user-{i:07d}" for i in range(20_000)]
    rows = []
    for name, hasher in (("hash64", hash64), ("clumping_hash", clumping_hash)):
        uniformity = chi_square_uniformity(values, n_buckets, hasher, seed)
        avalanche = avalanche_matrix(hasher, n_samples=n_avalanche_samples, seed=seed)
        rows.append(
            {
                "hash": name,
                "chi_square": uniformity.statistic,
                "p_value": uniformity.p_value,
                "max_over_expected": uniformity.max_over_expected,
                "is_uniform": uniformity.is_uniform,
                "avalanche_mean": avalanche.mean_flip_rate,
                "avalanche_worst_deviation": avalanche.worst_deviation,
                "avalanches": avalanche.avalanches,
            }
        )
    return pd.DataFrame(rows)


def bloom_error_curve(
    expected_items: int = 50_000,
    fp_targets: Sequence[float] = (0.1, 0.05, 0.01, 0.001, 0.0001),
    n_queries: int = 50_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Fill a filter to capacity at each target rate and measure what it actually does.

    The false-negative column is the one that matters: it is zero in every row, by construction,
    and would be a bug in the implementation if it were not.
    """
    added = [f"item-{i:09d}" for i in range(expected_items)]
    absent = [f"absent-{i:09d}" for i in range(n_queries)]
    rows = []
    for target in fp_targets:
        bloom = BloomFilter(expected_items, target, seed=seed)
        bloom.add_many(added)
        false_positives = int(bloom.contains_many(absent).sum())
        false_negatives = int((~bloom.contains_many(added)).sum())
        rows.append(
            {
                "target_fp_rate": target,
                "n_bits": bloom.n_bits,
                "n_hashes": bloom.n_hashes,
                "bits_per_item": bloom.n_bits / expected_items,
                "memory_bytes": bloom.memory_bytes(),
                "fill_ratio": bloom.fill_ratio,
                "theoretical_fp_rate": bloom.theoretical_fp_rate(),
                "measured_fp_rate": false_positives / n_queries,
                "false_negatives": false_negatives,
            }
        )
    return pd.DataFrame(rows)


def exact_set_bytes_per_item(n_sample: int = 20_000, seed: int = 0) -> float:
    """Measure what one item costs in a real Python ``set``, to project honestly from.

    A set of 9-character ids costs roughly 100 bytes an item once the hash table's own slots and
    the string objects are counted. That measured number is what the billion-item projection
    scales up - no hand-waving, and no allocating a billion strings to find out.
    """
    items = generate_unique_items(n_sample, seed=seed)
    exact: set[str] = set(items)
    total = sys.getsizeof(exact) + sum(sys.getsizeof(item) for item in exact)
    return total / n_sample


def bloom_memory_projection(
    scales: Sequence[int] = (1_000_000, 10_000_000, 100_000_000, 1_000_000_000),
    target_fp_rate: float = 0.01,
    bytes_per_exact_item: float | None = None,
) -> pd.DataFrame:
    """Project exact-set memory against Bloom memory at scales too large to allocate.

    Nothing is run here: the Bloom side is the closed-form sizing, and the exact side is a
    measured per-item cost multiplied out. Notebook 03 states exactly that when it shows the
    billion-item row.
    """
    if bytes_per_exact_item is None:
        bytes_per_exact_item = exact_set_bytes_per_item()
    rows = []
    for n_items in scales:
        geometry = sizing(n_items, target_fp_rate)
        exact_bytes = bytes_per_exact_item * n_items
        rows.append(
            {
                "n_items": n_items,
                "target_fp_rate": target_fp_rate,
                "exact_set_bytes": exact_bytes,
                "bloom_bytes": geometry.memory_bytes,
                "bits_per_item": geometry.bits_per_item,
                "savings_factor": exact_bytes / geometry.memory_bytes,
                "measured": False,
            }
        )
    return pd.DataFrame(rows)


def hyperloglog_error_curve(
    precisions: Sequence[int] = (8, 10, 12, 14),
    cardinalities: Sequence[int] = (1_000, 10_000, 100_000),
    seed: int = 0,
) -> pd.DataFrame:
    """Estimate versus truth across precisions and true cardinalities.

    The point of the table is that ``relative_error`` stays inside ``standard_error`` regardless
    of cardinality: the memory is fixed, and so is the accuracy.
    """
    rows = []
    for cardinality in cardinalities:
        items = generate_unique_items(cardinality, seed=seed)
        for precision in precisions:
            sketch = HyperLogLog(precision=precision, seed=seed)
            sketch.add_many(items)
            estimate = sketch.estimate()
            rows.append(
                {
                    "precision": precision,
                    "n_registers": sketch.m,
                    "memory_bytes": sketch.memory_bytes(),
                    "packed_memory_bytes": sketch.packed_memory_bytes(),
                    "true_cardinality": cardinality,
                    "estimate": estimate,
                    "relative_error": (estimate - cardinality) / cardinality,
                    "standard_error": standard_error(sketch.m),
                    "within_2_sigma": abs(estimate - cardinality) / cardinality
                    <= 2 * standard_error(sketch.m),
                }
            )
    return pd.DataFrame(rows)


def count_min_error_report(
    stream: Stream | None = None,
    epsilon: float = 0.001,
    delta: float = 0.01,
    top_k: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """Estimated versus true frequencies for the heaviest items, with the guaranteed bound.

    ``overshoot`` is never negative (the sketch cannot underestimate) and sits below
    ``error_bound`` for all but a ``delta`` fraction of items.
    """
    if stream is None:
        stream = generate_stream(n_items=200_000, n_unique=20_000, seed=seed)
    sketch = CountMinSketch(epsilon=epsilon, delta=delta, seed=seed)
    sketch.add_many(stream.items)
    bound = sketch.error_bound()
    rows = []
    for item, true_count in stream.top_k(top_k):
        estimate = sketch.estimate(item)
        rows.append(
            {
                "item": item,
                "true_count": true_count,
                "estimate": estimate,
                "overshoot": estimate - true_count,
                "error_bound": bound,
                "within_bound": (estimate - true_count) <= bound,
                "memory_bytes": sketch.memory_bytes(),
            }
        )
    return pd.DataFrame(rows)


def _sets_with_jaccard(target: float, set_size: int, seed: int) -> tuple[set[str], set[str]]:
    """Two sets of ``set_size`` elements whose exact Jaccard is as close to ``target`` as integers
    allow: overlap ``c`` satisfies ``J = c / (2n - c)``."""
    overlap = int(round(2 * set_size * target / (1.0 + target)))
    overlap = min(set_size, max(0, overlap))
    left = {f"s{seed}-a-{i}" for i in range(set_size)}
    shared = {f"s{seed}-a-{i}" for i in range(overlap)}
    right = shared | {f"s{seed}-b-{i}" for i in range(set_size - overlap)}
    return left, right


def minhash_error_curve(
    similarities: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 0.95),
    num_perm: int = 128,
    set_size: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Estimated Jaccard versus true Jaccard, with the sampling error the signature length allows."""
    hasher = MinHasher(num_perm=num_perm, seed=seed)
    rows = []
    for target in similarities:
        left, right = _sets_with_jaccard(target, set_size, seed)
        true_similarity = jaccard(left, right)
        estimate = hasher.estimated_jaccard(hasher.signature(left), hasher.signature(right))
        expected_std = float(
            np.sqrt(max(true_similarity * (1 - true_similarity), 1e-12) / num_perm)
        )
        rows.append(
            {
                "true_jaccard": true_similarity,
                "estimated_jaccard": estimate,
                "abs_error": abs(estimate - true_similarity),
                "expected_std": expected_std,
                "within_3_sigma": abs(estimate - true_similarity) <= 3 * expected_std + 1e-9,
                "signature_bytes": hasher.memory_bytes(),
            }
        )
    return pd.DataFrame(rows)


def lsh_recall_report(
    corpus: Corpus,
    bandings: Sequence[tuple[int, int]] = ((16, 8), (32, 4), (64, 2)),
    shingle_size: int = 5,
    similarity_threshold: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """Recall and work saved for several band/row splits of the same signatures.

    Ground truth is the exhaustive all-pairs Jaccard over shingle sets, so recall here is a real
    measurement rather than a proxy.
    """
    shingle_sets = [shingles(document, shingle_size) for document in corpus.documents]
    truth = {
        (i, j)
        for i in range(corpus.n_documents)
        for j in range(i + 1, corpus.n_documents)
        if jaccard(shingle_sets[i], shingle_sets[j]) >= similarity_threshold
    }
    rows = []
    for num_bands, rows_per_band in bandings:
        num_perm = num_bands * rows_per_band
        hasher = MinHasher(num_perm=num_perm, seed=seed)
        index = MinHashLSH(num_bands=num_bands, rows_per_band=rows_per_band, seed=seed)
        for i, shingle_set in enumerate(shingle_sets):
            index.insert(i, hasher.signature(shingle_set))
        candidates = {tuple(sorted(pair)) for pair in index.candidate_pairs()}
        found = candidates & truth
        stats = index.stats()
        rows.append(
            {
                "num_bands": num_bands,
                "rows_per_band": rows_per_band,
                "num_perm": num_perm,
                "threshold": approximate_threshold(num_bands, rows_per_band),
                "true_pairs": len(truth),
                "candidate_pairs": len(candidates),
                "all_pairs": corpus.n_all_pairs,
                "work_ratio": stats.work_ratio,
                "recall": len(found) / len(truth) if truth else float("nan"),
                "precision": len(found) / len(candidates) if candidates else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def feature_hashing_collision_curve(
    bucket_counts: Sequence[int] = (16, 32, 64, 128, 256, 512, 1024, 4096),
    n_tokens: int = 5_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Measured collision rate against the balls-in-bins prediction, per bucket count."""
    tokens = [f"user_id=user-{i:06d}" for i in range(n_tokens)]
    rows = []
    for n_buckets in bucket_counts:
        hasher = FeatureHasher(n_buckets=n_buckets, seed=seed)
        rows.append(
            {
                "n_buckets": n_buckets,
                "n_tokens": n_tokens,
                "measured_collision_rate": hasher.collision_rate(tokens),
                "expected_collision_rate": expected_collision_rate(n_tokens, n_buckets),
                "bytes_per_row": n_buckets * np.dtype(np.float64).itemsize,
            }
        )
    return pd.DataFrame(rows)


def _ridge_r2(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    ridge_alpha: float = 1.0,
) -> float:
    """Closed-form ridge regression, scored by R^2 on held-out rows (no sklearn needed)."""
    gram = train_x.T @ train_x + ridge_alpha * np.eye(train_x.shape[1])
    weights = np.linalg.solve(gram, train_x.T @ train_y)
    predictions = test_x @ weights
    residual = float(((test_y - predictions) ** 2).sum())
    total = float(((test_y - test_y.mean()) ** 2).sum())
    return 1.0 - residual / total if total > 0 else float("nan")


def feature_hashing_model_curve(
    table: pd.DataFrame,
    bucket_counts: Sequence[int] = (16, 32, 64, 128, 256, 512, 1024),
    max_rows: int = 5_000,
    ridge_alpha: float = 1.0,
    alternate_sign: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Model quality as a function of bucket count: the tradeoff, quantified.

    Too few buckets and collisions blur distinct categories into one column, so R^2 collapses;
    past the point where collisions are rare, extra buckets buy nothing but memory.

    ``alternate_sign=False`` turns off the signed-hash trick, so colliding categories accumulate
    instead of cancelling - the comparison notebook 06 draws.
    """
    sample = table.head(max_rows)
    rows_as_dicts = sample[["user_id", "city", "device"]].to_dict("records")
    target = sample["y"].to_numpy()
    split = int(0.75 * len(sample))
    results = []
    for n_buckets in bucket_counts:
        hasher = FeatureHasher(n_buckets=n_buckets, seed=seed, alternate_sign=alternate_sign)
        matrix = hasher.transform(rows_as_dicts)
        r2 = _ridge_r2(matrix[:split], target[:split], matrix[split:], target[split:], ridge_alpha)
        results.append(
            {
                "n_buckets": n_buckets,
                "alternate_sign": alternate_sign,
                "test_r2": r2,
                "collision_rate": hasher.collision_rate(
                    {f"user_id={value}" for value in sample["user_id"]}
                ),
                "memory_bytes": hasher.memory_bytes(len(sample)),
            }
        )
    return pd.DataFrame(results)
