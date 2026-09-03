"""HyperLogLog: count millions of unique items in a kilobyte, by watching for improbable luck.

The intuition before the formula. Flip a fair coin repeatedly and record the longest run of heads
you ever see. One head in a row is unremarkable; twenty in a row means you have been flipping for
roughly a million tries. The longest run is evidence about how many attempts there were, and it
costs nothing to remember - just one number.

Hash an item and you get a random-looking bit string, which is a sequence of coin flips. If some
item's hash starts with 20 zeros, you have probably seen about 2^20 distinct items (duplicates
hash to the same value, so they contribute nothing - which is exactly why this counts *uniques*).

One estimate is far too noisy, so HyperLogLog splits the stream: the first ``p`` bits of the hash
choose one of ``m = 2^p`` registers, each register keeps the longest zero-run it has seen, and the
harmonic mean across registers is the estimate (Flajolet et al., 2007)::

    E = alpha_m * m^2 / sum_j 2^(-M[j])          standard error ~ 1.04 / sqrt(m)

``p = 11`` gives 2048 registers, 1.5 KB when packed at 6 bits each, and 2.3% error - whether it is
counting a thousand uniques or a hundred million.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.core.hashes import hash64, hash_many, leading_zeros

MIN_PRECISION = 4
MAX_PRECISION = 18

# Register width in a packed implementation: a rank never exceeds 65 - p, which fits in 6 bits.
PACKED_REGISTER_BITS = 6

_SMALL_ALPHA = {16: 0.673, 32: 0.697, 64: 0.709}


def alpha(m: int) -> float:
    """The bias-correction constant for ``m`` registers (Flajolet et al., 2007)."""
    if m in _SMALL_ALPHA:
        return _SMALL_ALPHA[m]
    return 0.7213 / (1.0 + 1.079 / m)


def standard_error(m: int) -> float:
    """Relative standard error of the estimate: ``1.04 / sqrt(m)``."""
    return 1.04 / math.sqrt(m)


@dataclass(frozen=True)
class HyperLogLogSizing:
    """What a precision buys, before allocating anything."""

    precision: int
    n_registers: int
    packed_bytes: int
    standard_error: float


def sizing_for_error(target_error: float) -> HyperLogLogSizing:
    """Smallest precision whose standard error is at or below ``target_error``."""
    if not 0.0 < target_error < 1.0:
        raise ValueError("target_error must be in (0, 1)")
    for precision in range(MIN_PRECISION, MAX_PRECISION + 1):
        m = 1 << precision
        if standard_error(m) <= target_error:
            return HyperLogLogSizing(
                precision=precision,
                n_registers=m,
                packed_bytes=math.ceil(m * PACKED_REGISTER_BITS / 8),
                standard_error=standard_error(m),
            )
    raise ValueError(f"target_error {target_error} needs precision above {MAX_PRECISION}")


class HyperLogLog:
    """Cardinality estimation in fixed memory, whatever the true cardinality turns out to be.

    >>> hll = HyperLogLog(precision=11)
    >>> hll.add_many([f"user-{i}" for i in range(100_000)])
    >>> abs(hll.estimate() - 100_000) / 100_000 < 0.1
    True
    >>> hll.packed_memory_bytes()
    1536
    """

    def __init__(self, precision: int = 11, seed: int = 0) -> None:
        if not MIN_PRECISION <= precision <= MAX_PRECISION:
            raise ValueError(f"precision must be in {MIN_PRECISION}..{MAX_PRECISION}")
        self.precision = precision
        self.seed = seed
        self.m = 1 << precision
        self.rank_bits = 64 - precision
        self._registers = np.zeros(self.m, dtype=np.uint8)

    def __repr__(self) -> str:
        return f"HyperLogLog(precision={self.precision}, m={self.m})"

    @property
    def registers(self) -> np.ndarray:
        """A read-only view of the registers (notebook 04 plots their distribution)."""
        view = self._registers.view()
        view.flags.writeable = False
        return view

    def add(self, item: object) -> None:
        """Update the register the item's hash selects with the rank the rest of it shows."""
        value = hash64(item, self.seed)
        index = value >> self.rank_bits
        remainder = value & ((1 << self.rank_bits) - 1)
        rank = self.rank_bits - remainder.bit_length() + 1
        if rank > self._registers[index]:
            self._registers[index] = rank

    def add_many(self, items: Sequence[object], chunk_size: int = 200_000) -> None:
        """Add a batch. Chunked so a multi-million item stream never blows up memory."""
        n_items = len(items)
        for start in range(0, n_items, chunk_size):
            chunk = items[start : start + chunk_size]
            values = hash_many(chunk, self.seed)
            indices = (values >> np.uint64(self.rank_bits)).astype(np.int64)
            remainder = values & np.uint64((1 << self.rank_bits) - 1)
            ranks = (leading_zeros(remainder, self.rank_bits) + 1).astype(np.uint8)
            np.maximum.at(self._registers, indices, ranks)

    def n_zero_registers(self) -> int:
        """Registers still untouched - the signal the small-range correction uses."""
        return int(np.count_nonzero(self._registers == 0))

    def raw_estimate(self) -> float:
        """The harmonic-mean estimator, before any correction."""
        harmonic = float(np.sum(np.exp2(-self._registers.astype(np.float64))))
        return alpha(self.m) * self.m * self.m / harmonic

    def estimate(self) -> float:
        """Estimated number of distinct items seen.

        Two regimes, one correction:

        * **small range** - when few registers have been touched the harmonic estimator is biased
          low, and linear counting (``m ln(m / V)``, Whang et al., 1990) is exact enough;
        * **normal range** - the raw estimator.

        There is no large-range correction here because the hash is 64-bit: the original paper's
        correction exists only to undo collisions in a 32-bit hash space, which start to matter
        above ~4 billion distinct values. This implementation would need ~2^64 items to care.
        """
        raw = self.raw_estimate()
        zeros = self.n_zero_registers()
        if raw <= 2.5 * self.m and zeros > 0:
            return float(self.m * math.log(self.m / zeros))
        return raw

    def count(self) -> int:
        """:meth:`estimate` rounded to a whole number of items."""
        return int(round(self.estimate()))

    def merge(self, other: HyperLogLog) -> HyperLogLog:
        """Union of two sketches: the register-wise maximum. Exact, lossless, and O(m).

        This is why HyperLogLog is the shape of a distributed counter - shards can be counted
        separately and combined without ever exchanging the items themselves.
        """
        if (self.precision, self.seed) != (other.precision, other.seed):
            raise ValueError("sketches must share precision and seed to be merged")
        merged = HyperLogLog(self.precision, self.seed)
        merged._registers = np.maximum(self._registers, other._registers)
        return merged

    def standard_error(self) -> float:
        """Relative standard error of this sketch: ``1.04 / sqrt(m)``."""
        return standard_error(self.m)

    def error_band(self, sigmas: float = 2.0) -> float:
        """Relative error the estimate should stay inside with ~95% probability at 2 sigma."""
        return sigmas * self.standard_error()

    def memory_bytes(self) -> int:
        """Bytes this implementation actually holds (one byte per register, for clarity)."""
        return int(self._registers.nbytes)

    def packed_memory_bytes(self) -> int:
        """Bytes a production implementation needs: 6 bits per register.

        This is the number in the headline claim - 1536 bytes at ``p = 11``.
        """
        return math.ceil(self.m * PACKED_REGISTER_BITS / 8)
