"""MinHash must estimate the true Jaccard within the sampling error its signature length allows."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.similarity.minhash import MinHasher, jaccard, shingles

NUM_PERM = 128
SET_SIZE = 300


def _pair_with_jaccard(target: float, seed: int) -> tuple[set[str], set[str]]:
    overlap = int(round(2 * SET_SIZE * target / (1 + target)))
    left = {f"s{seed}-a-{i}" for i in range(SET_SIZE)}
    right = {f"s{seed}-a-{i}" for i in range(overlap)} | {
        f"s{seed}-b-{i}" for i in range(SET_SIZE - overlap)
    }
    return left, right


@pytest.mark.parametrize("target", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_estimate_matches_true_jaccard_within_sampling_error(target: float, seed: int) -> None:
    """Tolerance is 4 sqrt(J(1-J)/k) - the estimator's own standard error, not a tuned number."""
    left, right = _pair_with_jaccard(target, seed)
    truth = jaccard(left, right)
    hasher = MinHasher(num_perm=NUM_PERM, seed=seed)
    estimate = hasher.estimated_jaccard(hasher.signature(left), hasher.signature(right))
    tolerance = 4 * math.sqrt(truth * (1 - truth) / NUM_PERM) + 0.01
    assert abs(estimate - truth) <= tolerance


def test_identical_sets_agree_everywhere() -> None:
    hasher = MinHasher(num_perm=64, seed=0)
    items = {f"x{i}" for i in range(200)}
    assert hasher.estimated_jaccard(hasher.signature(items), hasher.signature(items)) == 1.0


def test_disjoint_sets_barely_agree() -> None:
    hasher = MinHasher(num_perm=128, seed=0)
    left = hasher.signature({f"a{i}" for i in range(200)})
    right = hasher.signature({f"b{i}" for i in range(200)})
    assert hasher.estimated_jaccard(left, right) < 0.05


def test_signature_is_order_independent_and_deterministic() -> None:
    hasher = MinHasher(num_perm=32, seed=3)
    items = [f"x{i}" for i in range(50)]
    assert np.array_equal(hasher.signature(items), hasher.signature(reversed(items)))
    assert np.array_equal(hasher.signature(items), MinHasher(32, seed=3).signature(items))


def test_signature_length_is_fixed_regardless_of_input_size() -> None:
    hasher = MinHasher(num_perm=128, seed=0)
    short = hasher.signature({"one", "two"})
    long = hasher.signature({f"x{i}" for i in range(20_000)})
    assert short.shape == long.shape == (128,)
    assert hasher.memory_bytes() == 128 * 8


def test_chunking_does_not_change_the_signature() -> None:
    hasher = MinHasher(num_perm=64, seed=1)
    items = [f"x{i}" for i in range(5_000)]
    assert np.array_equal(hasher.signature(items, chunk_size=64), hasher.signature(items))


def test_signature_matrix_stacks_rows() -> None:
    hasher = MinHasher(num_perm=16, seed=1)
    matrix = hasher.signature_matrix([{"a", "b"}, {"b", "c"}, {"z"}])
    assert matrix.shape == (3, 16)


def test_exact_jaccard() -> None:
    assert jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1}, set()) == 0.0


def test_word_and_char_shingles() -> None:
    text = "the quick brown fox jumps"
    assert shingles(text, 2) == {"the quick", "quick brown", "brown fox", "fox jumps"}
    assert "the q" in shingles(text, 5, kind="char")
    assert shingles("short", 10) == {"short"}
    assert shingles("", 3) == set()
    with pytest.raises(ValueError, match="kind"):
        shingles(text, 3, kind="bigram")
    with pytest.raises(ValueError, match="size"):
        shingles(text, 0)


def test_similar_texts_score_high_and_unrelated_texts_score_low() -> None:
    hasher = MinHasher(num_perm=128, seed=0)
    original = "probabilistic data structures trade accuracy for memory savings at scale"
    edited = "probabilistic data structures trade accuracy for memory savings at large scale"
    unrelated = "the price of tea in november depends on the weather and the harvest"
    signature = {
        name: hasher.signature(shingles(text, 3))
        for name, text in (("a", original), ("b", edited), ("c", unrelated))
    }
    assert hasher.estimated_jaccard(signature["a"], signature["b"]) > 0.5
    assert hasher.estimated_jaccard(signature["a"], signature["c"]) < 0.1


def test_invalid_num_perm_is_rejected() -> None:
    with pytest.raises(ValueError, match="num_perm"):
        MinHasher(num_perm=0)
