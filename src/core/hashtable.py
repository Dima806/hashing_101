"""The dictionary, from scratch, with both collision-resolution strategies.

Hashing turns "find this thing" into "compute where it must be". That is the whole reason a
``dict`` lookup is fast, and the whole reason collisions are the central problem: two keys that
compute the same slot have to share it somehow.

Two ways to share, both implemented here and compared in notebook 02:

* **chaining** - each slot holds a list; collisions extend the list;
* **open addressing** - collisions move to the next free slot (linear probing).

Both count probes, because probe count is what a lookup actually costs. Turn ``auto_resize`` off
and watch the cost stay flat and then explode as the load factor approaches 1 - that explosion is
why Python's dict resizes itself.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from src.core.hashes import hash64


@dataclass(frozen=True)
class ProbeStats:
    """What lookups cost, measured rather than assumed."""

    lookups: int
    probes: int
    mean_probes: float
    load_factor: float


class _BaseHashTable:
    """Shared bookkeeping: sizing, load factor, probe counting."""

    def __init__(
        self,
        capacity: int = 16,
        max_load_factor: float = 0.75,
        seed: int = 0,
        auto_resize: bool = True,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if not 0.0 < max_load_factor <= 1.0:
            raise ValueError("max_load_factor must be in (0, 1]")
        self.capacity = capacity
        self.max_load_factor = max_load_factor
        self.seed = seed
        self.auto_resize = auto_resize
        self.n_items = 0
        self.n_resizes = 0
        self._lookups = 0
        self._probes = 0

    def __len__(self) -> int:
        return self.n_items

    @property
    def load_factor(self) -> float:
        """Items per slot. Above ~0.75 the probe counts start to hurt."""
        return self.n_items / self.capacity

    def probe_stats(self) -> ProbeStats:
        """Probe counts accumulated since the last :meth:`reset_probe_stats`."""
        mean = self._probes / self._lookups if self._lookups else 0.0
        return ProbeStats(
            lookups=self._lookups,
            probes=self._probes,
            mean_probes=mean,
            load_factor=self.load_factor,
        )

    def reset_probe_stats(self) -> None:
        """Zero the probe counters (call before timing a specific workload)."""
        self._lookups = 0
        self._probes = 0

    def _slot(self, key: Any, capacity: int | None = None) -> int:
        return hash64(key, self.seed) % (capacity if capacity is not None else self.capacity)

    def _should_grow(self) -> bool:
        return self.auto_resize and (self.n_items + 1) / self.capacity > self.max_load_factor


class ChainingHashTable(_BaseHashTable):
    """Collisions share a slot by living in a list together.

    Lookup cost is the length of the chain, so it degrades gracefully: at load factor 4 a lookup
    walks about 4 entries, which is slow but never fails.
    """

    def __init__(
        self,
        capacity: int = 16,
        max_load_factor: float = 0.75,
        seed: int = 0,
        auto_resize: bool = True,
    ) -> None:
        super().__init__(capacity, max_load_factor, seed, auto_resize)
        self._buckets: list[list[tuple[Any, Any]]] = [[] for _ in range(capacity)]

    def __setitem__(self, key: Any, value: Any) -> None:
        if self._should_grow():
            self._resize(self.capacity * 2)
        chain = self._buckets[self._slot(key)]
        for i, (existing_key, _) in enumerate(chain):
            if existing_key == key:
                chain[i] = (key, value)
                return
        chain.append((key, value))
        self.n_items += 1

    def __getitem__(self, key: Any) -> Any:
        self._lookups += 1
        for probes, (existing_key, value) in enumerate(self._buckets[self._slot(key)], start=1):
            if existing_key == key:
                self._probes += probes
                return value
        self._probes += len(self._buckets[self._slot(key)]) or 1
        raise KeyError(key)

    def __delitem__(self, key: Any) -> None:
        chain = self._buckets[self._slot(key)]
        for i, (existing_key, _) in enumerate(chain):
            if existing_key == key:
                del chain[i]
                self.n_items -= 1
                return
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[Any]:
        for chain in self._buckets:
            for key, _ in chain:
                yield key

    def get(self, key: Any, default: Any = None) -> Any:
        """Value for ``key``, or ``default`` when it is absent."""
        try:
            return self[key]
        except KeyError:
            return default

    def items(self) -> Iterator[tuple[Any, Any]]:
        """Every (key, value) pair, in slot order."""
        for chain in self._buckets:
            yield from chain

    def bucket_lengths(self) -> list[int]:
        """Chain length per slot - the histogram notebook 02 plots."""
        return [len(chain) for chain in self._buckets]

    def _resize(self, new_capacity: int) -> None:
        old_items = list(self.items())
        self.capacity = new_capacity
        self._buckets = [[] for _ in range(new_capacity)]
        self.n_items = 0
        self.n_resizes += 1
        for key, value in old_items:
            self[key] = value

    def memory_bytes(self) -> int:
        """Approximate resident size: the slot array plus the chains and their entries."""
        total = sys.getsizeof(self._buckets)
        for chain in self._buckets:
            total += sys.getsizeof(chain)
            for entry in chain:
                total += sys.getsizeof(entry)
        return total


class _Empty:
    """Sentinel for a slot that was never used."""


class _Tombstone:
    """Sentinel for a slot whose entry was deleted (a probe must keep walking past it)."""


EMPTY = _Empty()
TOMBSTONE = _Tombstone()


class OpenAddressingHashTable(_BaseHashTable):
    """Collisions move to the next free slot (linear probing).

    Everything lives in one flat array, which is why it is cache-friendly and fast - right up
    until the table fills, when probe sequences grow without bound. Deletion leaves a tombstone,
    because simply emptying a slot would cut a probe sequence in half and lose keys behind it.
    """

    def __init__(
        self,
        capacity: int = 16,
        max_load_factor: float = 0.75,
        seed: int = 0,
        auto_resize: bool = True,
    ) -> None:
        super().__init__(capacity, max_load_factor, seed, auto_resize)
        self._keys: list[Any] = [EMPTY] * capacity
        self._values: list[Any] = [None] * capacity
        self.n_tombstones = 0

    def _probe_sequence(self, key: Any, capacity: int | None = None) -> Iterator[int]:
        size = capacity if capacity is not None else self.capacity
        start = self._slot(key, size)
        for step in range(size):
            yield (start + step) % size

    def __setitem__(self, key: Any, value: Any) -> None:
        if self._should_grow():
            self._resize(self.capacity * 2)
        first_free: int | None = None
        for slot in self._probe_sequence(key):
            current = self._keys[slot]
            if current is EMPTY:
                target = first_free if first_free is not None else slot
                self._keys[target] = key
                self._values[target] = value
                if first_free is not None:
                    self.n_tombstones -= 1
                self.n_items += 1
                return
            if current is TOMBSTONE:
                if first_free is None:
                    first_free = slot
                continue
            if current == key:
                self._values[slot] = value
                return
        if first_free is not None:
            self._keys[first_free] = key
            self._values[first_free] = value
            self.n_tombstones -= 1
            self.n_items += 1
            return
        raise RuntimeError("hash table is full; enable auto_resize or size it larger")

    def __getitem__(self, key: Any) -> Any:
        self._lookups += 1
        for probes, slot in enumerate(self._probe_sequence(key), start=1):
            current = self._keys[slot]
            if current is EMPTY:
                self._probes += probes
                raise KeyError(key)
            if current is not TOMBSTONE and current == key:
                self._probes += probes
                return self._values[slot]
        self._probes += self.capacity
        raise KeyError(key)

    def __delitem__(self, key: Any) -> None:
        for slot in self._probe_sequence(key):
            current = self._keys[slot]
            if current is EMPTY:
                break
            if current is not TOMBSTONE and current == key:
                self._keys[slot] = TOMBSTONE
                self._values[slot] = None
                self.n_items -= 1
                self.n_tombstones += 1
                return
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[Any]:
        for key in self._keys:
            if key is not EMPTY and key is not TOMBSTONE:
                yield key

    def get(self, key: Any, default: Any = None) -> Any:
        """Value for ``key``, or ``default`` when it is absent."""
        try:
            return self[key]
        except KeyError:
            return default

    def items(self) -> Iterator[tuple[Any, Any]]:
        """Every (key, value) pair, in slot order."""
        for key, value in zip(self._keys, self._values, strict=True):
            if key is not EMPTY and key is not TOMBSTONE:
                yield key, value

    def occupancy(self) -> list[int]:
        """1 for a live slot, 0 otherwise - the picture the collision visualiser draws."""
        return [0 if key is EMPTY or key is TOMBSTONE else 1 for key in self._keys]

    def _resize(self, new_capacity: int) -> None:
        old_items = list(self.items())
        self.capacity = new_capacity
        self._keys = [EMPTY] * new_capacity
        self._values = [None] * new_capacity
        self.n_items = 0
        self.n_tombstones = 0
        self.n_resizes += 1
        for key, value in old_items:
            self[key] = value

    def memory_bytes(self) -> int:
        """Approximate resident size: two flat arrays plus the live entries."""
        total = sys.getsizeof(self._keys) + sys.getsizeof(self._values)
        for key in self._keys:
            if key is not EMPTY and key is not TOMBSTONE:
                total += sys.getsizeof(key)
        return total
