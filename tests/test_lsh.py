"""LSH has to find the planted duplicates while looking at a tiny fraction of the pairs."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import Corpus
from src.similarity.lsh import (
    MinHashLSH,
    approximate_threshold,
    candidate_probability,
    choose_parameters,
)
from src.similarity.minhash import MinHasher, jaccard, shingles


def _index_corpus(corpus: Corpus, num_bands: int, rows_per_band: int, seed: int = 1):
    hasher = MinHasher(num_perm=num_bands * rows_per_band, seed=seed)
    index = MinHashLSH(num_bands=num_bands, rows_per_band=rows_per_band, seed=seed)
    shingle_sets = [shingles(document, 5) for document in corpus.documents]
    index.insert_many(list(range(corpus.n_documents)), hasher.signature_matrix(shingle_sets))
    return index, shingle_sets


def test_finds_the_planted_duplicates(corpus: Corpus) -> None:
    index, _ = _index_corpus(corpus, num_bands=32, rows_per_band=4)
    candidates = {tuple(sorted(pair)) for pair in index.candidate_pairs()}
    planted = {(min(a, b), max(a, b)) for a, b in corpus.duplicate_pairs}
    recall = len(candidates & planted) / len(planted)
    assert recall >= 0.85


def test_looks_at_a_fraction_of_the_pairs(corpus: Corpus) -> None:
    """The whole point: near-linear work instead of the quadratic all-pairs baseline."""
    index, _ = _index_corpus(corpus, num_bands=32, rows_per_band=4)
    stats = index.stats()
    assert stats.n_all_pairs == corpus.n_all_pairs
    assert stats.work_ratio < 0.02


def test_candidates_are_actually_similar(corpus: Corpus) -> None:
    index, shingle_sets = _index_corpus(corpus, num_bands=32, rows_per_band=4)
    similarities = [
        jaccard(shingle_sets[int(a)], shingle_sets[int(b)]) for a, b in index.candidate_pairs()
    ]
    assert np.mean(similarities) > index.threshold()


def test_higher_threshold_finds_fewer_pairs(corpus: Corpus) -> None:
    loose, _ = _index_corpus(corpus, num_bands=32, rows_per_band=4)
    strict, _ = _index_corpus(corpus, num_bands=8, rows_per_band=16)
    assert strict.threshold() > loose.threshold()
    assert len(strict.candidate_pairs()) <= len(loose.candidate_pairs())


def test_empirical_candidate_rate_matches_the_s_curve() -> None:
    """P(candidate | s) = 1 - (1 - s^r)^b, measured over pairs of known similarity."""
    num_bands, rows_per_band = 16, 4
    target = 0.6
    hits = 0
    trials = 60
    for seed in range(trials):
        size, overlap = 200, int(round(2 * 200 * target / (1 + target)))
        left = {f"s{seed}a{i}" for i in range(size)}
        right = {f"s{seed}a{i}" for i in range(overlap)} | {
            f"s{seed}b{i}" for i in range(size - overlap)
        }
        hasher = MinHasher(num_perm=num_bands * rows_per_band, seed=seed)
        index = MinHashLSH(num_bands=num_bands, rows_per_band=rows_per_band, seed=seed)
        index.insert("left", hasher.signature(left))
        hits += bool(index.query(hasher.signature(right)))
    predicted = candidate_probability(jaccard(left, right), num_bands, rows_per_band)
    sampling_error = float(np.sqrt(predicted * (1 - predicted) / trials))
    assert abs(hits / trials - predicted) <= 4 * sampling_error + 0.05


def test_query_can_exclude_itself() -> None:
    hasher = MinHasher(num_perm=32, seed=0)
    index = MinHashLSH(num_bands=8, rows_per_band=4, seed=0)
    signature = hasher.signature({f"x{i}" for i in range(50)})
    index.insert("doc", signature)
    assert index.query(signature) == {"doc"}
    assert index.query(signature, exclude="doc") == set()
    assert len(index) == 1


def test_threshold_and_probability_formulas() -> None:
    assert approximate_threshold(16, 8) == pytest.approx((1 / 16) ** (1 / 8))
    assert candidate_probability(0.0, 16, 8) == 0.0
    assert candidate_probability(1.0, 16, 8) == 1.0
    assert candidate_probability(0.5, 16, 8) == pytest.approx(1 - (1 - 0.5**8) ** 16)


def test_choose_parameters_respects_the_signature_length() -> None:
    bands, rows = choose_parameters(128, 0.5)
    assert bands * rows == 128
    assert abs(approximate_threshold(bands, rows) - 0.5) < 0.15
    with pytest.raises(ValueError, match="target_threshold"):
        choose_parameters(128, 1.5)


def test_signature_length_must_match_the_banding() -> None:
    index = MinHashLSH(num_bands=8, rows_per_band=4)
    with pytest.raises(ValueError, match="expected 32"):
        index.insert("doc", np.zeros(16, dtype=np.uint64))


def test_invalid_banding_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        MinHashLSH(num_bands=0, rows_per_band=4)


def test_memory_grows_with_the_index() -> None:
    hasher = MinHasher(num_perm=32, seed=0)
    index = MinHashLSH(num_bands=8, rows_per_band=4, seed=0)
    for i in range(10):
        index.insert(f"doc{i}", hasher.signature({f"x{i}-{j}" for j in range(20)}))
    assert index.memory_bytes() > 10 * 32 * 8
