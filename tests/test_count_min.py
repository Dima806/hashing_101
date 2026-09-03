"""Count-Min's guarantee is one-sided upwards: never below the truth, rarely far above it."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.data import Stream
from src.probabilistic.count_min import CountMinSketch, optimal_depth, optimal_width


def test_sizing_matches_the_published_formulas() -> None:
    assert optimal_width(0.001) == math.ceil(math.e / 0.001)
    assert optimal_depth(0.01) == math.ceil(math.log(1 / 0.01))
    sketch = CountMinSketch(epsilon=0.001, delta=0.01)
    assert sketch.memory_bytes() == sketch.width * sketch.depth * 8


def test_never_underestimates(stream: Stream) -> None:
    """The flagship property: collisions can only add counts, so the minimum is an upper bound."""
    sketch = CountMinSketch(epsilon=0.01, delta=0.05, seed=1)
    sketch.add_many(stream.items)
    items = list(stream.true_counts)[:2_000]
    estimates = sketch.estimate_many(items)
    truths = np.array([stream.true_counts[item] for item in items])
    assert bool((estimates >= truths).all())


def test_overshoot_stays_inside_the_bound_for_almost_every_item(stream: Stream) -> None:
    """``eps * N`` with probability ``1 - delta`` - so a few violations are allowed, not many."""
    delta = 0.05
    sketch = CountMinSketch(epsilon=0.01, delta=delta, seed=1)
    sketch.add_many(stream.items)
    items = list(stream.true_counts)[:2_000]
    overshoot = sketch.estimate_many(items) - np.array(
        [stream.true_counts[item] for item in items]
    )
    violations = float((overshoot > sketch.error_bound()).mean())
    assert violations <= 3 * delta


def test_heavy_hitters_are_found(stream: Stream) -> None:
    sketch = CountMinSketch(epsilon=0.001, delta=0.01, seed=1)
    sketch.add_many(stream.items)
    heaviest = [item for item, _ in stream.top_k(5)]
    found = [item for item, _ in sketch.heavy_hitters(stream.true_counts, fraction=0.01)]
    assert set(heaviest[:3]).issubset(found)


def test_scalar_and_batch_paths_agree(stream: Stream) -> None:
    scalar = CountMinSketch(epsilon=0.01, delta=0.1, seed=2)
    for item in stream.items[:5_000]:
        scalar.add(item)
    batch = CountMinSketch(epsilon=0.01, delta=0.1, seed=2)
    batch.add_many(stream.items[:5_000])
    assert np.array_equal(scalar._counters, batch._counters)
    assert scalar.total == batch.total


def test_weighted_updates() -> None:
    sketch = CountMinSketch(epsilon=0.01, delta=0.1, seed=1)
    sketch.add_many(["a", "b"], [10, 5])
    assert sketch.estimate("a") >= 10
    assert sketch.total == 15
    bounded = sketch.estimate_with_bound("a")
    assert bounded.lower_bound <= 10 <= bounded.estimate


def test_unseen_item_estimates_zero_on_an_empty_sketch() -> None:
    sketch = CountMinSketch(epsilon=0.01, delta=0.1, seed=1)
    assert sketch.estimate("never-added") == 0
    assert list(sketch.estimate_many([])) == []


@pytest.mark.parametrize(("epsilon", "delta"), [(0.0, 0.1), (1.0, 0.1), (0.1, 0.0), (0.1, 1.0)])
def test_invalid_parameters_are_rejected(epsilon: float, delta: float) -> None:
    with pytest.raises(ValueError):
        CountMinSketch(epsilon=epsilon, delta=delta)


def test_negative_counts_are_rejected() -> None:
    """The min estimator is only an upper bound while every update is non-negative."""
    sketch = CountMinSketch(epsilon=0.01, delta=0.1, seed=1)
    with pytest.raises(ValueError, match="non-negative"):
        sketch.add("a", -1)
    with pytest.raises(ValueError, match="non-negative"):
        sketch.add_many(["a"], [-1])
