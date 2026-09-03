"""A hash is good when it is uniform and it avalanches. Both are measurable; here they are."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.hashes import (
    MASK64,
    HashFamily,
    avalanche_matrix,
    chi_square_uniformity,
    clumping_hash,
    hash64,
    hash_many,
    leading_zeros,
    splitmix64,
    splitmix64_array,
    to_bytes,
)

VALUES = [f"user-{i:07d}" for i in range(20_000)]


@pytest.mark.parametrize("backend", ["auto", "python"])
def test_hash_is_deterministic_and_64_bit(backend: str) -> None:
    for value in ("hello", b"hello", 42, -1, 2**70):
        first = hash64(value, 3, backend=backend)
        assert first == hash64(value, 3, backend=backend)
        assert 0 <= first <= MASK64


def test_seed_changes_the_hash() -> None:
    assert hash64("hello", 0) != hash64("hello", 1)


def test_to_bytes_is_canonical() -> None:
    assert to_bytes("abc") == b"abc"
    assert to_bytes(b"abc") == b"abc"
    assert to_bytes(1) == (1).to_bytes(8, "little")
    assert to_bytes(np.int64(1)) == to_bytes(1)


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        hash64("x", backend="md5")


@pytest.mark.parametrize("backend", ["auto", "python"])
def test_good_hash_is_uniform(backend: str) -> None:
    result = chi_square_uniformity(
        VALUES, 256, lambda value, seed: hash64(value, seed, backend=backend)
    )
    assert result.is_uniform, f"chi-square p={result.p_value}"
    assert result.max_over_expected < 1.5


def test_bad_hash_is_not_uniform() -> None:
    """The sum of the input bytes clumps badly - which is the whole lesson of notebook 01."""
    result = chi_square_uniformity(VALUES, 256, clumping_hash)
    assert not result.is_uniform
    assert result.p_value < 1e-6


def test_good_hash_avalanches() -> None:
    result = avalanche_matrix(hash64, n_samples=400, seed=1)
    assert abs(result.mean_flip_rate - 0.5) < 0.02
    assert result.avalanches


def test_bad_hash_does_not_avalanche() -> None:
    result = avalanche_matrix(clumping_hash, n_samples=400, seed=1)
    assert result.mean_flip_rate < 0.1
    assert not result.avalanches


def test_splitmix_scalar_matches_vector() -> None:
    values = np.array([0, 1, 12345, 2**63, MASK64], dtype=np.uint64)
    expected = np.array([splitmix64(int(v)) for v in values], dtype=np.uint64)
    assert np.array_equal(splitmix64_array(values), expected)


def test_leading_zeros_matches_definition() -> None:
    values = np.array([0, 1, 2, 2**63, 2**40], dtype=np.uint64)
    assert list(leading_zeros(values, 64)) == [64, 63, 62, 0, 23]
    assert list(leading_zeros(np.array([0, 1], dtype=np.uint64), 8)) == [8, 7]


def test_leading_zeros_rejects_bad_width() -> None:
    with pytest.raises(ValueError, match="width"):
        leading_zeros(np.array([1], dtype=np.uint64), 65)


def test_hash_many_matches_hash64() -> None:
    batch = hash_many(VALUES[:500], 5)
    assert batch.dtype == np.uint64
    assert [int(v) for v in batch] == [hash64(v, 5) for v in VALUES[:500]]


def test_hash_family_batch_matches_scalar() -> None:
    """The batch path is an optimisation, not a different structure - it must agree exactly."""
    family = HashFamily(k=7, seed=3)
    m = 1_000_003
    scalar = np.array([family.indices(value, m) for value in VALUES[:300]], dtype=np.uint64).T
    assert np.array_equal(family.indices_many(VALUES[:300], m), scalar)


def test_hash_family_indices_are_in_range() -> None:
    family = HashFamily(k=5, seed=1)
    indices = family.indices("anything", 64)
    assert len(indices) == 5
    assert all(0 <= index < 64 for index in indices)


def test_hash_family_rejects_empty_family() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        HashFamily(k=0)
