"""Bloom filter: a bit array plus k hashes, and one guarantee that never bends.

To add an item, set the k bits its hashes point to. To test membership, check whether all k bits
are set. That is the entire structure.

The consequence is a **one-sided error**: a bit is only set by an item that was added, so if any
of the k bits is clear the item was definitely never added. ``item not in filter`` is therefore
always the truth, and ``item in filter`` is "probably, with a false-positive rate you chose when
you sized the filter". That asymmetry is exactly what a deduplication filter needs, and it is
asserted in ``tests/test_bloom.py``: zero false negatives, ever.

Sizing (Bloom, 1970), for ``n`` expected items and target false-positive rate ``p``::

    m = -n * ln(p) / (ln 2)^2      bits
    k = (m / n) * ln 2             hashes
    fp(m, k, n) = (1 - e^(-k n / m))^k
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from src.core.hashes import HashFamily

LN2 = math.log(2.0)


def optimal_n_bits(expected_items: int, target_fp_rate: float) -> int:
    """``m = -n ln(p) / (ln 2)^2``, rounded up to a whole bit."""
    _validate_sizing(expected_items, target_fp_rate)
    return max(1, math.ceil(-expected_items * math.log(target_fp_rate) / (LN2**2)))


def optimal_n_hashes(n_bits: int, expected_items: int) -> int:
    """``k = (m / n) ln 2``, the count that minimises the false-positive rate."""
    return max(1, round((n_bits / expected_items) * LN2))


def false_positive_rate(n_bits: int, n_hashes: int, n_items: int) -> float:
    """``(1 - e^(-k n / m))^k`` - the rate this geometry gives after ``n`` insertions."""
    if n_items <= 0:
        return 0.0
    return float((1.0 - math.exp(-n_hashes * n_items / n_bits)) ** n_hashes)


def _validate_sizing(expected_items: int, target_fp_rate: float) -> None:
    if expected_items < 1:
        raise ValueError("expected_items must be positive")
    if not 0.0 < target_fp_rate < 1.0:
        raise ValueError("target_fp_rate must be in (0, 1)")


@dataclass(frozen=True)
class BloomSizing:
    """The geometry a target error implies, before any memory is allocated.

    Used to project the headline claim honestly: a billion items at 1% needs ~1.2 GB of bits,
    at 2% needs ~1 GB, and the exact ``set`` needs tens of gigabytes - all computed, not run.
    """

    expected_items: int
    target_fp_rate: float
    n_bits: int
    n_hashes: int

    @property
    def memory_bytes(self) -> int:
        """Bytes the bit array occupies."""
        return (self.n_bits + 7) // 8

    @property
    def bits_per_item(self) -> float:
        """The number that makes the trade visible: ~9.6 bits per item at 1%, whatever n is."""
        return self.n_bits / self.expected_items

    @property
    def achieved_fp_rate(self) -> float:
        """False-positive rate at design capacity, after rounding m and k to integers."""
        return false_positive_rate(self.n_bits, self.n_hashes, self.expected_items)


def sizing(expected_items: int, target_fp_rate: float) -> BloomSizing:
    """Size a filter for ``expected_items`` at ``target_fp_rate`` without allocating it."""
    n_bits = optimal_n_bits(expected_items, target_fp_rate)
    return BloomSizing(
        expected_items=expected_items,
        target_fp_rate=target_fp_rate,
        n_bits=n_bits,
        n_hashes=optimal_n_hashes(n_bits, expected_items),
    )


class BloomFilter:
    """A Bloom filter sized from what you actually know: how many items, and what error you accept.

    >>> bloom = BloomFilter(expected_items=10_000, target_fp_rate=0.01)
    >>> bloom.add("event-1")
    >>> "event-1" in bloom          # never wrong in this direction
    True
    >>> bloom.memory_bytes() < 13_000
    True
    """

    def __init__(
        self,
        expected_items: int,
        target_fp_rate: float = 0.01,
        seed: int = 0,
        n_bits: int | None = None,
        n_hashes: int | None = None,
    ) -> None:
        _validate_sizing(expected_items, target_fp_rate)
        derived = sizing(expected_items, target_fp_rate)
        self.expected_items = expected_items
        self.target_fp_rate = target_fp_rate
        self.n_bits = n_bits if n_bits is not None else derived.n_bits
        self.n_hashes = n_hashes if n_hashes is not None else derived.n_hashes
        self.seed = seed
        self.n_added = 0
        self._bits = np.zeros((self.n_bits + 7) // 8, dtype=np.uint8)
        self._family = HashFamily(k=self.n_hashes, seed=seed)

    def __repr__(self) -> str:
        return (
            f"BloomFilter(expected_items={self.expected_items}, "
            f"target_fp_rate={self.target_fp_rate}, n_bits={self.n_bits}, "
            f"n_hashes={self.n_hashes})"
        )

    def __len__(self) -> int:
        return self.n_added

    def add(self, item: object) -> None:
        """Set the k bits of ``item``."""
        for index in self._family.indices(item, self.n_bits):
            self._bits[index >> 3] |= np.uint8(1 << (index & 7))
        self.n_added += 1

    def add_many(self, items: Sequence[object]) -> None:
        """Add a batch - same bits as calling :meth:`add` in a loop, one pass of numpy work."""
        if len(items) == 0:
            return
        indices = self._family.indices_many(items, self.n_bits).reshape(-1)
        byte_index = (indices >> np.uint64(3)).astype(np.int64)
        bit_mask = np.uint8(1) << (indices & np.uint64(7)).astype(np.uint8)
        np.bitwise_or.at(self._bits, byte_index, bit_mask)
        self.n_added += len(items)

    def __contains__(self, item: object) -> bool:
        """True if all k bits are set. False means *definitely not added* - the whole point."""
        for index in self._family.indices(item, self.n_bits):
            if not self._bits[index >> 3] & np.uint8(1 << (index & 7)):
                return False
        return True

    def contains_many(self, items: Sequence[object]) -> np.ndarray:
        """Vectorised membership test for a batch, as a boolean array."""
        if len(items) == 0:
            return np.zeros(0, dtype=bool)
        indices = self._family.indices_many(items, self.n_bits)
        byte_index = (indices >> np.uint64(3)).astype(np.int64)
        bit_mask = np.uint8(1) << (indices & np.uint64(7)).astype(np.uint8)
        return np.asarray((self._bits[byte_index] & bit_mask).astype(bool).all(axis=0))

    @property
    def bits_set(self) -> int:
        """How many of the m bits are set (popcount over the byte array)."""
        return int(np.unpackbits(self._bits).sum())

    @property
    def fill_ratio(self) -> float:
        """Fraction of bits set. The filter is saturated - and useless - as this nears 1."""
        return self.bits_set / self.n_bits

    def estimated_fp_rate(self) -> float:
        """False-positive rate implied by the bits actually set: ``(bits_set / m)^k``."""
        return float(self.fill_ratio**self.n_hashes)

    def theoretical_fp_rate(self, n_items: int | None = None) -> float:
        """``(1 - e^(-k n / m))^k`` for ``n`` items (default: however many were added)."""
        return false_positive_rate(
            self.n_bits, self.n_hashes, self.n_added if n_items is None else n_items
        )

    def memory_bytes(self) -> int:
        """Bytes of bit array. Compare with the ``set`` in ``evaluation/comparison.py``."""
        return int(self._bits.nbytes)

    def union(self, other: BloomFilter) -> BloomFilter:
        """Bitwise OR of two filters with identical geometry (a merge of two streams)."""
        if (self.n_bits, self.n_hashes, self.seed) != (other.n_bits, other.n_hashes, other.seed):
            raise ValueError("filters must share geometry and seed to be merged")
        merged = BloomFilter(
            self.expected_items,
            self.target_fp_rate,
            self.seed,
            n_bits=self.n_bits,
            n_hashes=self.n_hashes,
        )
        merged._bits = self._bits | other._bits
        merged.n_added = self.n_added + other.n_added
        return merged


def measure_false_positive_rate(
    bloom: BloomFilter, absent_items: Iterable[object]
) -> tuple[float, int]:
    """Query items known to be absent; return (measured rate, number of queries).

    The measurement notebook 03 plots against the theoretical curve.
    """
    items = list(absent_items)
    if not items:
        raise ValueError("need at least one absent item to measure a rate")
    hits = int(bloom.contains_many(items).sum())
    return hits / len(items), len(items)
