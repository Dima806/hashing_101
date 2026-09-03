"""Hash functions, and the two diagnostics that decide whether one is any good.

A hash is only ever asked to do two things:

* **uniformity** - spread inputs evenly across buckets, so no bucket is overloaded;
* **avalanche** - flip one input bit and about half the output bits change, so inputs that look
  alike do not land near each other.

"Random-looking but deterministic" is the whole requirement. Every structure in this project
inherits its error guarantees from those two properties, which is why a bad hash quietly destroys
all of them - see :func:`clumping_hash` for what bad looks like.

Backends: ``mmh3`` (MurmurHash3, a fast C extension) when installed, otherwise a pure-Python
FNV-1a + SplitMix64 fallback so the notebooks run anywhere. Both are seeded and deterministic.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1

# FNV-1a 64-bit parameters.
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3

# SplitMix64 mixing constants (Steele et al., 2014). Used as the finaliser of the pure-Python
# fallback and as the second hash of the double-hashing scheme below.
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_MIX_1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MIX_2 = 0x94D049BB133111EB

_mmh3 = importlib.import_module("mmh3") if importlib.util.find_spec("mmh3") is not None else None

HAS_MMH3 = _mmh3 is not None
"""Whether the fast MurmurHash3 backend is available (a pure-Python fallback runs if not)."""

Backend = str  # "auto" | "mmh3" | "python"


def to_bytes(value: object) -> bytes:
    """Canonical byte encoding of a hashable value.

    Integers are encoded as 64-bit two's complement (so ``-1`` and ``2**64 - 1`` collide, which is
    fine and documented); wider integers get as many bytes as they need.
    """
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bool, int, np.integer)):
        as_int = int(value)
        if -(2**63) <= as_int < 2**64:
            return (as_int & MASK64).to_bytes(8, "little")
        n_bytes = (as_int.bit_length() + 8) // 8
        return as_int.to_bytes(n_bytes, "little", signed=True)
    return repr(value).encode("utf-8")


def splitmix64(x: int) -> int:
    """SplitMix64 finaliser: turns a merely-different integer into a random-looking one."""
    z = (x + _SPLITMIX_GAMMA) & MASK64
    z = ((z ^ (z >> 30)) * _SPLITMIX_MIX_1) & MASK64
    z = ((z ^ (z >> 27)) * _SPLITMIX_MIX_2) & MASK64
    return z ^ (z >> 31)


def splitmix64_array(values: np.ndarray) -> np.ndarray:
    """Vectorised :func:`splitmix64` (unsigned 64-bit arithmetic wraps, exactly as intended)."""
    z = values.astype(np.uint64) + np.uint64(_SPLITMIX_GAMMA)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(_SPLITMIX_MIX_1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(_SPLITMIX_MIX_2)
    return z ^ (z >> np.uint64(31))


def _fnv1a64(data: bytes, seed: int) -> int:
    """FNV-1a: one multiply and one xor per byte. Cheap, and poor on its own."""
    acc = (_FNV_OFFSET ^ splitmix64(seed)) & MASK64
    for byte in data:
        acc = ((acc ^ byte) * _FNV_PRIME) & MASK64
    return acc


def hash64(value: object, seed: int = 0, *, backend: Backend = "auto") -> int:
    """Hash any value to an unsigned 64-bit integer.

    ``backend="auto"`` uses mmh3 when installed and the pure-Python fallback otherwise; pass
    ``"python"`` to force the fallback (the tests check both behave).
    """
    data = to_bytes(value)
    if backend not in ("auto", "mmh3", "python"):
        raise ValueError(f"unknown backend {backend!r}")
    if backend == "mmh3" and _mmh3 is None:
        raise RuntimeError("mmh3 backend requested but mmh3 is not installed")
    if _mmh3 is not None and backend in ("auto", "mmh3"):
        return int(_mmh3.hash128(data, seed, signed=False)) & MASK64
    # FNV-1a alone has weak avalanche, so it is finalised through SplitMix64.
    return splitmix64(_fnv1a64(data, seed))


def hash_many(values: Iterable[object], seed: int = 0, *, backend: Backend = "auto") -> np.ndarray:
    """Hash a batch of values to a ``uint64`` array.

    One Python-level loop over the (C-speed) hash, then every structure does its arithmetic on the
    whole array at once - that is what keeps million-item streams to seconds on 2 CPUs.
    """
    return np.fromiter(
        (hash64(value, seed, backend=backend) for value in values),
        dtype=np.uint64,
        count=len(values) if isinstance(values, Sequence) else -1,
    )


def clumping_hash(value: object, seed: int = 0) -> int:
    """A deliberately bad hash: the sum of the input bytes.

    Deterministic, cheap, and useless - ``"user_100"`` and ``"user_010"`` land in the same bucket,
    and the sums of similar strings cluster in a narrow range instead of spreading. Notebook 01
    plots this next to :func:`hash64` to show what "uniform" buys you.
    """
    del seed  # a bad hash ignores its seed too
    return sum(to_bytes(value))


def leading_zeros(values: np.ndarray, width: int = 64) -> np.ndarray:
    """Count leading zero bits of each value within a field of ``width`` bits.

    All-zero values return ``width``. This is the primitive HyperLogLog counts with.
    """
    if not 1 <= width <= 64:
        raise ValueError("width must be in 1..64")
    flat = np.ascontiguousarray(np.asarray(values, dtype=np.uint64)).reshape(-1)
    if flat.size == 0:
        return np.zeros(0, dtype=np.int64)
    as_bytes = flat.view(np.uint8).reshape(-1, 8)
    if sys.byteorder == "little":
        as_bytes = as_bytes[:, ::-1]
    bits = np.unpackbits(np.ascontiguousarray(as_bytes), axis=1)[:, 64 - width :]
    nonzero = bits.any(axis=1)
    return np.where(nonzero, bits.argmax(axis=1), width).astype(np.int64)


@dataclass(frozen=True)
class HashFamily:
    """``k`` independent-enough hash functions from one, by double hashing.

    ``h_i(x) = h1(x) + i * h2(x)`` (mod 2**64, then mod m). Kirsch and Mitzenmacher (2006) showed
    this costs nothing in false-positive rate versus ``k`` separate hashes, and it is why a Bloom
    filter with k=7 still only hashes each item once.
    """

    k: int
    seed: int = 0
    backend: Backend = "auto"

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be at least 1")

    def base(self, value: object) -> int:
        """The single 64-bit hash the whole family is derived from."""
        return hash64(value, self.seed, backend=self.backend)

    def indices(self, value: object, m: int) -> list[int]:
        """The ``k`` bucket indices of ``value`` in a table of ``m`` slots."""
        h1 = self.base(value)
        h2 = splitmix64(h1) | 1  # odd, so the stride is coprime with any power-of-two m
        return [((h1 + i * h2) & MASK64) % m for i in range(self.k)]

    def indices_many(self, values: Iterable[object], m: int) -> np.ndarray:
        """Bucket indices for a batch: shape ``(k, n)``, identical to :meth:`indices` per item."""
        h1 = hash_many(values, self.seed, backend=self.backend)
        h2 = splitmix64_array(h1) | np.uint64(1)
        steps = np.arange(self.k, dtype=np.uint64).reshape(-1, 1)
        return (h1.reshape(1, -1) + steps * h2.reshape(1, -1)) % np.uint64(m)


@dataclass(frozen=True)
class UniformityResult:
    """Chi-square goodness-of-fit of hashed values against a uniform bucket distribution."""

    counts: np.ndarray
    statistic: float
    p_value: float
    dof: int
    max_over_expected: float

    @property
    def is_uniform(self) -> bool:
        """True when the test fails to reject uniformity at the 1% level."""
        return self.p_value > 0.01


def bucket_counts(
    values: Iterable[object],
    n_buckets: int,
    hasher: Callable[[object, int], int] = hash64,
    seed: int = 0,
) -> np.ndarray:
    """How many of ``values`` land in each of ``n_buckets`` buckets."""
    counts = np.zeros(n_buckets, dtype=np.int64)
    for value in values:
        counts[hasher(value, seed) % n_buckets] += 1
    return counts


def chi_square_uniformity(
    values: Iterable[object],
    n_buckets: int,
    hasher: Callable[[object, int], int] = hash64,
    seed: int = 0,
) -> UniformityResult:
    """Test whether a hash spreads ``values`` evenly across ``n_buckets``.

    A good hash gives a large p-value (no evidence of clumping); :func:`clumping_hash` gives a
    p-value indistinguishable from zero.
    """
    counts = bucket_counts(values, n_buckets, hasher, seed)
    total = int(counts.sum())
    if total == 0:
        raise ValueError("no values to test")
    expected = total / n_buckets
    result = stats.chisquare(counts)
    return UniformityResult(
        counts=counts,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        dof=n_buckets - 1,
        max_over_expected=float(counts.max() / expected),
    )


@dataclass(frozen=True)
class AvalancheResult:
    """Per-bit output flip rates when a single input bit is flipped."""

    matrix: np.ndarray  # (64 input bits, 64 output bits) of flip probabilities
    mean_flip_rate: float
    worst_deviation: float  # largest |rate - 0.5| over the whole matrix
    n_samples: int

    @property
    def avalanches(self) -> bool:
        """True when every input bit flips every output bit close to half the time.

        The tolerance is the sampling noise of the measurement itself: each cell is a mean of
        ``n_samples`` coin flips, so 5 standard errors is the band a fair hash stays inside.
        """
        return self.worst_deviation <= 5.0 * float(np.sqrt(0.25 / self.n_samples))


def avalanche_matrix(
    hasher: Callable[[object, int], int] = hash64,
    n_samples: int = 1000,
    seed: int = 0,
) -> AvalancheResult:
    """Measure the avalanche effect of a hash over random 64-bit inputs.

    For each of the 64 input bits: flip it, hash before and after, and record which output bits
    changed. A good hash sits at 0.5 everywhere; :func:`clumping_hash` barely moves.
    """
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, 1 << 63, size=n_samples, dtype=np.int64).astype(np.uint64)
    matrix = np.zeros((64, 64), dtype=np.float64)
    base = np.fromiter(
        (hasher(int(value), 0) & MASK64 for value in samples), dtype=np.uint64, count=n_samples
    )
    for bit in range(64):
        flipped_inputs = samples ^ np.uint64(1 << bit)
        flipped = np.fromiter(
            (hasher(int(value), 0) & MASK64 for value in flipped_inputs),
            dtype=np.uint64,
            count=n_samples,
        )
        difference = base ^ flipped
        as_bytes = np.ascontiguousarray(difference).view(np.uint8).reshape(-1, 8)
        if sys.byteorder == "little":
            as_bytes = as_bytes[:, ::-1]
        bits = np.unpackbits(np.ascontiguousarray(as_bytes), axis=1)
        matrix[bit] = bits.mean(axis=0)
    return AvalancheResult(
        matrix=matrix,
        mean_flip_rate=float(matrix.mean()),
        worst_deviation=float(np.abs(matrix - 0.5).max()),
        n_samples=n_samples,
    )
