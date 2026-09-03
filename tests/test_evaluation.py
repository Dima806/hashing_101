"""The evaluation harness is what the notebooks plot, so its invariants are worth asserting."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import Corpus, Stream
from src.evaluation.comparison import (
    compare_cardinality,
    compare_membership,
    compare_near_duplicate_search,
    decision_guide,
    deep_sizeof,
    memory_summary,
)
from src.evaluation.error_analysis import (
    bloom_error_curve,
    bloom_memory_projection,
    count_min_error_report,
    exact_set_bytes_per_item,
    feature_hashing_collision_curve,
    feature_hashing_model_curve,
    hash_quality_report,
    hyperloglog_error_curve,
    lsh_recall_report,
    minhash_error_curve,
)
from src.probabilistic.hyperloglog import standard_error


def test_hash_quality_report_separates_good_from_bad() -> None:
    report = hash_quality_report(
        values=[f"user-{i}" for i in range(3_000)], n_buckets=64, n_avalanche_samples=200
    )
    good = report[report["hash"] == "hash64"].iloc[0]
    bad = report[report["hash"] == "clumping_hash"].iloc[0]
    assert good["is_uniform"] and good["avalanches"]
    assert not bad["is_uniform"] and not bad["avalanches"]


def test_bloom_error_curve_never_reports_a_false_negative() -> None:
    curve = bloom_error_curve(expected_items=5_000, fp_targets=(0.1, 0.01), n_queries=5_000)
    assert (curve["false_negatives"] == 0).all()
    assert (curve["measured_fp_rate"] <= curve["target_fp_rate"] * 2).all()
    assert curve["memory_bytes"].is_monotonic_increasing  # tighter target costs more memory


def test_bloom_projection_is_labelled_as_a_projection() -> None:
    """A billion items is computed, not allocated - and the table says so."""
    projection = bloom_memory_projection(scales=(1_000_000, 1_000_000_000), target_fp_rate=0.01)
    assert not projection["measured"].any()
    billion = projection.iloc[-1]
    assert billion["savings_factor"] > 50
    assert billion["bloom_bytes"] < 2e9  # a couple of gigabytes of bits, not tens
    assert billion["bits_per_item"] == pytest.approx(9.585, abs=0.01)


def test_exact_set_cost_is_measured_in_a_plausible_range() -> None:
    assert 50 < exact_set_bytes_per_item(n_sample=5_000) < 200


def test_hyperloglog_curve_stays_inside_the_band() -> None:
    curve = hyperloglog_error_curve(precisions=(10, 12), cardinalities=(5_000, 50_000))
    assert (curve["relative_error"].abs() <= 4 * curve["standard_error"]).all()
    assert (curve["packed_memory_bytes"] < 13_000).all()
    for precision, group in curve.groupby("precision"):
        assert group["memory_bytes"].nunique() == 1  # memory does not grow with cardinality
        assert group["standard_error"].iloc[0] == pytest.approx(
            standard_error(int(group["n_registers"].iloc[0]))
        )
        assert precision in (10, 12)


def test_count_min_report_is_one_sided(stream: Stream) -> None:
    report = count_min_error_report(stream, epsilon=0.01, delta=0.05, top_k=10)
    assert (report["overshoot"] >= 0).all()
    assert report["within_bound"].all()


def test_minhash_curve_tracks_the_truth() -> None:
    curve = minhash_error_curve(similarities=(0.3, 0.7), num_perm=128, set_size=200, seed=1)
    assert (curve["abs_error"] <= 4 * curve["expected_std"] + 0.02).all()


def test_lsh_report_trades_a_little_recall_for_a_lot_of_work(corpus: Corpus) -> None:
    report = lsh_recall_report(corpus, bandings=((32, 4),), similarity_threshold=0.5)
    row = report.iloc[0]
    assert row["recall"] >= 0.85
    assert row["work_ratio"] < 0.02
    assert row["candidate_pairs"] < row["all_pairs"] / 50


def test_feature_hashing_collision_curve_matches_theory() -> None:
    curve = feature_hashing_collision_curve(bucket_counts=(64, 512, 4096), n_tokens=2_000)
    assert (
        (curve["measured_collision_rate"] - curve["expected_collision_rate"]).abs() < 0.03
    ).all()
    assert curve["measured_collision_rate"].is_monotonic_decreasing


def test_feature_hashing_model_curve_shows_the_tradeoff(categorical_table: pd.DataFrame) -> None:
    """Too few buckets and collisions destroy the signal; enough buckets and the model recovers."""
    curve = feature_hashing_model_curve(
        categorical_table, bucket_counts=(8, 512), max_rows=2_000, seed=1
    )
    assert curve["test_r2"].iloc[-1] > curve["test_r2"].iloc[0]
    assert curve["collision_rate"].iloc[0] > curve["collision_rate"].iloc[-1]


def test_membership_comparison(stream: Stream) -> None:
    frame = compare_membership(n_items=5_000, target_fp_rate=0.01, n_queries=5_000)
    exact, bloom = frame.iloc[0], frame.iloc[1]
    assert exact["false_positive_rate"] == 0.0
    assert bloom["false_negatives"] == 0
    assert bloom["memory_bytes"] < exact["memory_bytes"] / 10
    assert bloom["false_positive_rate"] < 0.05


def test_cardinality_comparison() -> None:
    frame = compare_cardinality(n_unique=20_000, precision=11)
    exact, sketch = frame.iloc[0], frame.iloc[1]
    assert sketch["memory_bytes"] == 1_536
    assert sketch["memory_bytes"] < exact["memory_bytes"] / 100
    assert abs(sketch["relative_error"]) <= 4 * standard_error(2_048)


def test_near_duplicate_comparison(corpus: Corpus) -> None:
    frame = compare_near_duplicate_search(corpus, num_bands=32, rows_per_band=4)
    exhaustive, lsh = frame.iloc[0], frame.iloc[1]
    assert lsh["comparisons"] < exhaustive["comparisons"] / 50
    assert lsh["recall"] >= 0.85


def test_decision_guide_covers_every_structure() -> None:
    guide = decision_guide()
    assert len(guide) == 7
    structures = " ".join(guide["structure"])
    for name in ("Bloom", "HyperLogLog", "Count-Min", "MinHash", "Feature hashing"):
        assert name in structures


def test_memory_helpers() -> None:
    assert deep_sizeof([f"item-{i}" for i in range(100)]) > 100
    summary = memory_summary([("bloom", 1_000_000, 10_000)])
    assert summary["savings_factor"].iloc[0] == pytest.approx(100.0)
