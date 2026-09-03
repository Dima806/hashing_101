"""Counting Bloom: deletion, without ever opening the door to a false negative."""

from __future__ import annotations

import pytest

from src.probabilistic.counting_bloom import CountingBloomFilter

ITEMS = [f"item-{i:06d}" for i in range(2_000)]


def test_no_false_negatives_for_items_still_present() -> None:
    """The Bloom guarantee survives deletion: whatever is still in the filter still tests true."""
    counting = CountingBloomFilter(5_000, 0.01, seed=1)
    for item in ITEMS:
        counting.add(item)
    for item in ITEMS[:500]:
        counting.remove(item)
    assert all(item in counting for item in ITEMS[500:])


def test_remove_undoes_add() -> None:
    counting = CountingBloomFilter(1_000, 0.01, seed=1)
    counting.add("only-item")
    assert "only-item" in counting
    counting.remove("only-item")
    assert "only-item" not in counting
    assert len(counting) == 0


def test_removing_an_item_that_was_never_added_is_refused() -> None:
    """Silently decrementing would corrupt the filter into reporting false negatives."""
    counting = CountingBloomFilter(1_000, 0.01, seed=1)
    counting.add("present")
    with pytest.raises(KeyError, match="never added"):
        counting.remove("absent")


def test_counters_saturate_instead_of_wrapping() -> None:
    """A wrapped counter would go to zero and delete an item that is still there."""
    counting = CountingBloomFilter(100, 0.01, seed=1, counter_bits=8)
    for _ in range(400):
        counting.add("hot-item")
    assert counting.n_saturated > 0
    assert "hot-item" in counting
    assert counting.count_estimate("hot-item") == 255


def test_memory_is_the_price_of_deletion() -> None:
    counting = CountingBloomFilter(10_000, 0.01, seed=1, counter_bits=8)
    assert counting.memory_bytes() == counting.n_slots
    assert counting.memory_bytes() == pytest.approx(8 * (counting.n_slots / 8), rel=0.01)


def test_error_rate_matches_the_plain_filter() -> None:
    counting = CountingBloomFilter(2_000, 0.01, seed=1)
    counting.add_many(ITEMS)
    assert counting.theoretical_fp_rate() == pytest.approx(0.01, abs=0.005)
    assert counting.estimated_fp_rate() == pytest.approx(counting.theoretical_fp_rate(), rel=0.3)


def test_counter_width_is_validated() -> None:
    with pytest.raises(ValueError, match="counter_bits"):
        CountingBloomFilter(100, 0.01, counter_bits=7)
