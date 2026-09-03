"""Counting Bloom filter: the same idea with counters instead of bits, so items can be removed.

A plain Bloom filter cannot delete: clearing an item's bits would also clear bits that other items
rely on, and *that* would create false negatives. Replace each bit with a small counter, increment
on add and decrement on remove, and deletion becomes safe - at the cost of 8x (or 16x) the memory.

Two rules keep the one-sided guarantee intact:

* a **saturated** counter (one that hit its maximum) is never decremented, because its true value
  is unknown - decrementing it could take a live item's counter to zero;
* removing an item that was never added is a caller error; it can corrupt the filter into
  reporting false negatives, so :meth:`CountingBloomFilter.remove` refuses when any counter of the
  item is already zero.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.core.hashes import HashFamily
from src.probabilistic.bloom import false_positive_rate, sizing

_COUNTER_DTYPES: dict[int, type[np.unsignedinteger]] = {8: np.uint8, 16: np.uint16}


class CountingBloomFilter:
    """A Bloom filter whose slots count, so :meth:`remove` is well defined."""

    def __init__(
        self,
        expected_items: int,
        target_fp_rate: float = 0.01,
        seed: int = 0,
        counter_bits: int = 8,
    ) -> None:
        if counter_bits not in _COUNTER_DTYPES:
            raise ValueError(f"counter_bits must be one of {sorted(_COUNTER_DTYPES)}")
        derived = sizing(expected_items, target_fp_rate)
        self.expected_items = expected_items
        self.target_fp_rate = target_fp_rate
        self.n_slots = derived.n_bits
        self.n_hashes = derived.n_hashes
        self.counter_bits = counter_bits
        self.seed = seed
        self.n_added = 0
        self.n_saturated = 0
        self._dtype = _COUNTER_DTYPES[counter_bits]
        self._max_count = int(np.iinfo(self._dtype).max)
        self._counters = np.zeros(self.n_slots, dtype=self._dtype)
        self._family = HashFamily(k=self.n_hashes, seed=seed)

    def __repr__(self) -> str:
        return (
            f"CountingBloomFilter(expected_items={self.expected_items}, "
            f"target_fp_rate={self.target_fp_rate}, n_slots={self.n_slots}, "
            f"n_hashes={self.n_hashes}, counter_bits={self.counter_bits})"
        )

    def __len__(self) -> int:
        return self.n_added

    def add(self, item: object) -> None:
        """Increment the k counters of ``item``, saturating rather than wrapping."""
        for index in self._family.indices(item, self.n_slots):
            if self._counters[index] < self._max_count:
                self._counters[index] += 1
            else:
                self.n_saturated += 1
        self.n_added += 1

    def add_many(self, items: Sequence[object]) -> None:
        """Add a batch. Counters saturate at the dtype maximum, never wrap to zero."""
        for item in items:
            self.add(item)

    def remove(self, item: object) -> None:
        """Decrement the k counters of ``item``.

        Raises ``KeyError`` if any counter is already zero: the item was never added (or was
        already removed), and decrementing anyway is how a counting filter starts lying.
        """
        indices = self._family.indices(item, self.n_slots)
        if any(self._counters[index] == 0 for index in indices):
            raise KeyError(f"{item!r} was never added; refusing to corrupt the filter")
        for index in indices:
            if self._counters[index] < self._max_count:
                self._counters[index] -= 1
        self.n_added -= 1

    def __contains__(self, item: object) -> bool:
        """True when every counter of ``item`` is non-zero (same one-sided error as Bloom)."""
        return all(self._counters[index] > 0 for index in self._family.indices(item, self.n_slots))

    def count_estimate(self, item: object) -> int:
        """The minimum counter over the item's slots - a crude upper-bounded frequency."""
        return int(
            min(self._counters[index] for index in self._family.indices(item, self.n_slots))
        )

    @property
    def fill_ratio(self) -> float:
        """Fraction of counters above zero."""
        return float(np.count_nonzero(self._counters) / self.n_slots)

    def estimated_fp_rate(self) -> float:
        """False-positive rate implied by the non-zero counters."""
        return float(self.fill_ratio**self.n_hashes)

    def theoretical_fp_rate(self, n_items: int | None = None) -> float:
        """``(1 - e^(-k n / m))^k`` - identical to the plain filter, which is the point."""
        return false_positive_rate(
            self.n_slots, self.n_hashes, self.n_added if n_items is None else n_items
        )

    def memory_bytes(self) -> int:
        """Bytes of counters: ``counter_bits / 1`` times a plain Bloom filter of the same shape."""
        return int(self._counters.nbytes)
