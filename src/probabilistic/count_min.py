"""Count-Min Sketch: how many times did each item appear, in memory that does not grow.

A Bloom filter answers "have I seen this?"; Count-Min answers "how often?". Same primitive: a
grid of ``d`` rows by ``w`` columns of counters, one hash per row. To add an item, increment one
counter in every row. To estimate its count, take the **minimum** across rows - collisions can
only ever add someone else's counts, so the smallest cell is the least contaminated.

That gives the guarantee (Cormode and Muthukrishnan, 2005), for a stream of total weight ``N``::

    estimate >= true count                     always, for non-negative counts
    estimate <= true count + epsilon * N       with probability at least 1 - delta

    w = ceil(e / epsilon)      columns
    d = ceil(ln(1 / delta))    rows

The error is one-sided *upwards*, which is the mirror image of the Bloom filter, and it is why
Count-Min is the standard tool for finding heavy hitters in a stream: an item the sketch says is
rare really is rare.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from src.core.hashes import HashFamily


def optimal_width(epsilon: float) -> int:
    """``w = ceil(e / epsilon)`` columns for an additive error of ``epsilon * N``."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    return max(1, math.ceil(math.e / epsilon))


def optimal_depth(delta: float) -> int:
    """``d = ceil(ln(1 / delta))`` rows for a failure probability of ``delta``."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    return max(1, math.ceil(math.log(1.0 / delta)))


@dataclass(frozen=True)
class CountMinEstimate:
    """An estimate together with the bound it is guaranteed to sit inside."""

    item: object
    estimate: int
    error_bound: float

    @property
    def lower_bound(self) -> float:
        """The true count is at least this (the sketch never underestimates)."""
        return max(0.0, self.estimate - self.error_bound)


class CountMinSketch:
    """Frequency estimation for a stream, in ``d * w`` counters.

    >>> sketch = CountMinSketch(epsilon=0.001, delta=0.01)
    >>> sketch.add("popular", 500)
    >>> sketch.estimate("popular") >= 500      # never underestimates
    True
    """

    def __init__(self, epsilon: float = 0.001, delta: float = 0.01, seed: int = 0) -> None:
        self.epsilon = epsilon
        self.delta = delta
        self.seed = seed
        self.width = optimal_width(epsilon)
        self.depth = optimal_depth(delta)
        self.total = 0
        self._counters = np.zeros((self.depth, self.width), dtype=np.int64)
        self._family = HashFamily(k=self.depth, seed=seed)
        self._rows = np.arange(self.depth)

    def __repr__(self) -> str:
        return (
            f"CountMinSketch(epsilon={self.epsilon}, delta={self.delta}, "
            f"width={self.width}, depth={self.depth})"
        )

    def add(self, item: object, count: int = 1) -> None:
        """Add ``count`` occurrences of ``item``."""
        if count < 0:
            raise ValueError("count must be non-negative; the min estimator assumes it")
        columns = self._family.indices(item, self.width)
        for row, column in enumerate(columns):
            self._counters[row, column] += count
        self.total += count

    def add_many(self, items: Sequence[object], counts: Sequence[int] | None = None) -> None:
        """Add a batch of items, optionally with per-item counts."""
        if len(items) == 0:
            return
        columns = self._family.indices_many(items, self.width).astype(np.int64)
        weights = (
            np.ones(len(items), dtype=np.int64)
            if counts is None
            else np.asarray(counts, dtype=np.int64)
        )
        if (weights < 0).any():
            raise ValueError("counts must be non-negative; the min estimator assumes it")
        for row in range(self.depth):
            np.add.at(self._counters[row], columns[row], weights)
        self.total += int(weights.sum())

    def estimate(self, item: object) -> int:
        """Minimum counter across rows: an upper bound on the true count, never below it."""
        columns = self._family.indices(item, self.width)
        return int(min(self._counters[row, column] for row, column in enumerate(columns)))

    def estimate_many(self, items: Sequence[object]) -> np.ndarray:
        """Vectorised :meth:`estimate` for a batch."""
        if len(items) == 0:
            return np.zeros(0, dtype=np.int64)
        columns = self._family.indices_many(items, self.width).astype(np.int64)
        return self._counters[self._rows[:, None], columns].min(axis=0)

    def estimate_with_bound(self, item: object) -> CountMinEstimate:
        """Estimate plus the ``epsilon * N`` slack it is guaranteed to fall within."""
        return CountMinEstimate(
            item=item, estimate=self.estimate(item), error_bound=self.error_bound()
        )

    def error_bound(self) -> float:
        """``epsilon * N``: the most the estimate can overshoot, with probability 1 - delta."""
        return self.epsilon * self.total

    def heavy_hitters(self, candidates: Iterable[object], fraction: float = 0.01) -> list[tuple]:
        """Candidates whose estimated count is at least ``fraction`` of the whole stream.

        Count-Min cannot enumerate items (it stores none), so heavy-hitter detection always takes
        a candidate list - in a real pipeline, the items seen in the current window.
        """
        found = [(item, self.estimate(item)) for item in candidates]
        cutoff = fraction * self.total
        return sorted(
            [(item, count) for item, count in found if count >= cutoff],
            key=lambda pair: pair[1],
            reverse=True,
        )

    def memory_bytes(self) -> int:
        """Bytes of counters - fixed, whatever the stream does."""
        return int(self._counters.nbytes)
