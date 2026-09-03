"""HyperLogLog is judged against its theoretical error band, never against a point value."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import generate_unique_items
from src.probabilistic.hyperloglog import (
    HyperLogLog,
    alpha,
    sizing_for_error,
    standard_error,
)


def test_alpha_matches_the_published_constants() -> None:
    assert alpha(16) == 0.673
    assert alpha(32) == 0.697
    assert alpha(64) == 0.709
    assert alpha(2048) == pytest.approx(0.7213 / (1 + 1.079 / 2048))


def test_precision_11_is_the_headline_claim() -> None:
    """2048 registers, 1.5 KB packed, ~2.3% standard error."""
    sketch = HyperLogLog(precision=11)
    assert sketch.m == 2048
    assert sketch.packed_memory_bytes() == 1536
    assert sketch.standard_error() == pytest.approx(0.023, abs=0.001)


@pytest.mark.parametrize(
    ("precision", "cardinality"),
    [(10, 20_000), (12, 50_000), (14, 100_000)],
)
def test_estimate_lands_inside_the_theoretical_band(precision: int, cardinality: int) -> None:
    """The tolerance is 4 standard errors of the estimator itself - derived, not tuned."""
    sketch = HyperLogLog(precision=precision, seed=1)
    sketch.add_many(generate_unique_items(cardinality, seed=cardinality))
    relative_error = abs(sketch.estimate() - cardinality) / cardinality
    assert relative_error <= 4 * standard_error(sketch.m)


def test_accuracy_holds_as_cardinality_grows_while_memory_does_not() -> None:
    sketch = HyperLogLog(precision=12, seed=1)
    errors = []
    for cardinality in (10_000, 100_000, 400_000):
        fresh = HyperLogLog(precision=12, seed=1)
        fresh.add_many(generate_unique_items(cardinality, seed=cardinality))
        errors.append(abs(fresh.estimate() - cardinality) / cardinality)
        assert fresh.memory_bytes() == sketch.memory_bytes()
    assert max(errors) <= 4 * standard_error(sketch.m)


def test_small_range_correction_handles_tiny_cardinalities() -> None:
    """Without linear counting the raw estimator is badly biased when most registers are empty."""
    sketch = HyperLogLog(precision=12, seed=1)
    items = generate_unique_items(200, seed=3)
    sketch.add_many(items)
    assert sketch.n_zero_registers() > 0
    assert abs(sketch.estimate() - 200) / 200 < 0.05


def test_duplicates_do_not_move_the_estimate() -> None:
    """It counts uniques: a duplicate hashes to the same place and updates nothing."""
    items = generate_unique_items(5_000, seed=5)
    once = HyperLogLog(precision=12, seed=1)
    once.add_many(items)
    twice = HyperLogLog(precision=12, seed=1)
    twice.add_many(items * 3)
    assert once.estimate() == twice.estimate()


def test_scalar_and_batch_paths_agree() -> None:
    items = generate_unique_items(2_000, seed=9)
    scalar = HyperLogLog(precision=10, seed=1)
    for item in items:
        scalar.add(item)
    batch = HyperLogLog(precision=10, seed=1)
    batch.add_many(items)
    assert np.array_equal(scalar.registers, batch.registers)


def test_merge_is_a_union() -> None:
    """Sketches merge without the items ever meeting - why HyperLogLog fits distributed counting."""
    items = generate_unique_items(20_000, seed=11)
    left = HyperLogLog(precision=12, seed=1)
    left.add_many(items[:10_000])
    right = HyperLogLog(precision=12, seed=1)
    right.add_many(items[5_000:])
    merged = left.merge(right)
    whole = HyperLogLog(precision=12, seed=1)
    whole.add_many(items)
    assert np.array_equal(merged.registers, whole.registers)
    assert merged.estimate() == whole.estimate()


def test_merge_rejects_mismatched_sketches() -> None:
    with pytest.raises(ValueError, match="precision and seed"):
        HyperLogLog(precision=10).merge(HyperLogLog(precision=11))


def test_registers_are_read_only() -> None:
    sketch = HyperLogLog(precision=8)
    with pytest.raises(ValueError):
        sketch.registers[0] = 1


def test_sizing_for_error_picks_the_smallest_sufficient_precision() -> None:
    chosen = sizing_for_error(0.02)
    assert chosen.standard_error <= 0.02
    assert standard_error(1 << (chosen.precision - 1)) > 0.02
    assert chosen.packed_bytes == chosen.n_registers * 6 // 8


def test_invalid_precision_is_rejected() -> None:
    with pytest.raises(ValueError, match="precision"):
        HyperLogLog(precision=3)
    with pytest.raises(ValueError, match="precision"):
        HyperLogLog(precision=19)
