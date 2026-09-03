"""The Bloom filter's contract: never a false negative, and a false-positive rate you asked for."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.probabilistic.bloom import (
    BloomFilter,
    false_positive_rate,
    measure_false_positive_rate,
    optimal_n_bits,
    optimal_n_hashes,
    sizing,
)

N_ITEMS = 20_000
ITEMS = [f"item-{i:07d}" for i in range(N_ITEMS)]
ABSENT = [f"absent-{i:07d}" for i in range(20_000)]


def test_sizing_matches_the_published_formulas() -> None:
    n, p = 100_000, 0.01
    assert optimal_n_bits(n, p) == math.ceil(-n * math.log(p) / math.log(2) ** 2)
    assert optimal_n_hashes(optimal_n_bits(n, p), n) == 7
    geometry = sizing(n, p)
    # ~9.6 bits per item at 1%, whatever n is - the number that makes the trade obvious.
    assert geometry.bits_per_item == pytest.approx(9.585, abs=0.01)
    assert geometry.memory_bytes == (geometry.n_bits + 7) // 8


@pytest.mark.parametrize("target", [0.1, 0.01, 0.001])
def test_no_false_negatives_ever(target: float) -> None:
    """The flagship assertion of the whole project: 'not in' is always the truth."""
    bloom = BloomFilter(N_ITEMS, target, seed=1)
    bloom.add_many(ITEMS)
    assert int((~bloom.contains_many(ITEMS)).sum()) == 0
    assert all(item in bloom for item in ITEMS[::500])


def test_no_false_negatives_even_when_overfilled() -> None:
    """Past capacity the error rate degrades - but only ever in the false-positive direction."""
    bloom = BloomFilter(1_000, 0.01, seed=1)
    bloom.add_many(ITEMS)
    assert int((~bloom.contains_many(ITEMS)).sum()) == 0
    assert bloom.fill_ratio > 0.9


@pytest.mark.parametrize("target", [0.1, 0.05, 0.01])
def test_measured_false_positive_rate_matches_theory(target: float) -> None:
    """Tolerance is the sampling error of the measurement plus rounding of m and k, not a guess."""
    bloom = BloomFilter(N_ITEMS, target, seed=1)
    bloom.add_many(ITEMS)
    measured, n_queries = measure_false_positive_rate(bloom, ABSENT)
    predicted = bloom.theoretical_fp_rate()
    sampling_error = math.sqrt(predicted * (1 - predicted) / n_queries)
    assert abs(measured - predicted) <= 4 * sampling_error + 0.1 * predicted


def test_estimated_rate_from_fill_tracks_theory() -> None:
    bloom = BloomFilter(N_ITEMS, 0.01, seed=1)
    bloom.add_many(ITEMS)
    assert bloom.estimated_fp_rate() == pytest.approx(bloom.theoretical_fp_rate(), rel=0.1)


def test_scalar_and_batch_paths_agree() -> None:
    one = BloomFilter(1_000, 0.01, seed=4)
    other = BloomFilter(1_000, 0.01, seed=4)
    for item in ITEMS[:1_000]:
        one.add(item)
    other.add_many(ITEMS[:1_000])
    assert np.array_equal(one._bits, other._bits)
    assert np.array_equal(one.contains_many(ABSENT[:500]), other.contains_many(ABSENT[:500]))
    assert [item in one for item in ABSENT[:500]] == list(other.contains_many(ABSENT[:500]))


def test_memory_is_exactly_the_bit_array() -> None:
    bloom = BloomFilter(N_ITEMS, 0.01, seed=1)
    assert bloom.memory_bytes() == (bloom.n_bits + 7) // 8
    # Two orders of magnitude below an exact set of the same items (~100 bytes each).
    assert bloom.memory_bytes() < N_ITEMS * 100 / 50


def test_empty_filter_reports_nothing_present() -> None:
    bloom = BloomFilter(1_000, 0.01, seed=1)
    assert not bloom.contains_many(ABSENT[:1_000]).any()
    assert bloom.fill_ratio == 0.0
    assert bloom.theoretical_fp_rate() == 0.0
    bloom.add_many([])
    assert len(bloom) == 0


def test_union_merges_two_streams() -> None:
    left = BloomFilter(1_000, 0.01, seed=2)
    right = BloomFilter(1_000, 0.01, seed=2)
    left.add_many(ITEMS[:500])
    right.add_many(ITEMS[500:1_000])
    merged = left.union(right)
    assert int((~merged.contains_many(ITEMS[:1_000])).sum()) == 0
    with pytest.raises(ValueError, match="geometry"):
        left.union(BloomFilter(2_000, 0.01, seed=2))


def test_false_positive_rate_formula() -> None:
    assert false_positive_rate(1_000, 3, 0) == 0.0
    assert false_positive_rate(9_585, 7, 1_000) == pytest.approx(
        (1 - math.exp(-7 * 1_000 / 9_585)) ** 7
    )


@pytest.mark.parametrize(
    ("expected_items", "target"),
    [(0, 0.01), (100, 0.0), (100, 1.0), (100, -0.1)],
)
def test_invalid_sizing_is_rejected(expected_items: int, target: float) -> None:
    with pytest.raises(ValueError):
        BloomFilter(expected_items, target)
