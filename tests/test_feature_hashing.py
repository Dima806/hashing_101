"""Feature hashing: fixed width, no vocabulary, and collisions that behave the way theory says."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.feature_hashing import FeatureHasher, expected_collision_rate

TOKENS = [f"user_id=user-{i:06d}" for i in range(2_000)]


def test_output_shape_is_fixed_by_the_bucket_count() -> None:
    hasher = FeatureHasher(n_buckets=64, seed=0)
    rows = [{"city": "berlin", "device": "ios"}, {"city": "lisbon", "device": "web"}]
    assert hasher.transform(rows).shape == (2, 64)
    assert hasher.transform([]).shape == (0, 64)


def test_transform_is_deterministic_and_needs_no_vocabulary() -> None:
    """Unseen categories at serving time simply land somewhere - nothing to update, nothing to break."""
    hasher = FeatureHasher(n_buckets=32, seed=0)
    row = [{"city": "berlin"}]
    assert np.array_equal(hasher.transform(row), hasher.transform(row))
    unseen = hasher.transform([{"city": "a-city-never-seen-before"}])
    assert unseen.shape == (1, 32)
    assert np.count_nonzero(unseen) == 1


@pytest.mark.parametrize("n_buckets", [64, 256, 1024, 4096])
def test_measured_collision_rate_matches_balls_in_bins(n_buckets: int) -> None:
    hasher = FeatureHasher(n_buckets=n_buckets, seed=0)
    measured = hasher.collision_rate(TOKENS)
    predicted = expected_collision_rate(len(TOKENS), n_buckets)
    assert abs(measured - predicted) < 0.03


def test_fewer_buckets_means_more_collisions() -> None:
    rates = [FeatureHasher(n_buckets=n, seed=0).collision_rate(TOKENS) for n in (32, 256, 4096)]
    assert rates[0] > rates[1] > rates[2]
    assert rates[0] > 0.9  # 2,000 categories into 32 buckets: almost everything collides


def test_colliding_groups_name_the_categories_that_share_a_column() -> None:
    hasher = FeatureHasher(n_buckets=16, seed=0)
    groups = hasher.colliding_groups(TOKENS[:100])
    assert groups
    assert all(len(members) > 1 for members in groups.values())
    assert all(hasher.bucket_of(member) == bucket for bucket, m in groups.items() for member in m)


def test_signed_hashing_cancels_collisions_in_expectation() -> None:
    hasher = FeatureHasher(n_buckets=256, seed=0, alternate_sign=True)
    signs = np.array([hasher.sign_of(token) for token in TOKENS])
    assert set(np.unique(signs)) == {-1, 1}
    # Mean sign should sit within a few standard errors of zero: sqrt(1/n) = 0.022 here.
    assert abs(signs.mean()) < 4 / np.sqrt(len(TOKENS))


def test_unsigned_hashing_only_adds() -> None:
    hasher = FeatureHasher(n_buckets=8, seed=0, alternate_sign=False)
    vector = hasher.transform_row([f"token-{i}" for i in range(50)])
    assert (vector >= 0).all()
    assert vector.sum() == pytest.approx(50.0)


def test_numeric_values_are_used_as_weights() -> None:
    hasher = FeatureHasher(n_buckets=16, seed=0, alternate_sign=False)
    vector = hasher.transform_row({"spend": 12.5})
    assert vector.sum() == pytest.approx(12.5)


def test_feature_names_keep_values_distinct() -> None:
    """``city=berlin`` and ``device=berlin`` must not be the same feature."""
    hasher = FeatureHasher(n_buckets=1024, seed=0)
    assert hasher.bucket_of("city=berlin") != hasher.bucket_of("device=berlin")


def test_memory_is_fixed_width() -> None:
    hasher = FeatureHasher(n_buckets=128, seed=0)
    assert hasher.memory_bytes(1_000) == 1_000 * 128 * 8


def test_expected_collision_rate_edges() -> None:
    assert expected_collision_rate(0, 16) == 0.0
    assert expected_collision_rate(1, 16) == pytest.approx(0.0, abs=1e-12)
    assert expected_collision_rate(10_000, 8) > 0.99


def test_invalid_bucket_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_buckets"):
        FeatureHasher(n_buckets=0)
