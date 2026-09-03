"""Figures are a deliverable, so they get a smoke test: they build, and they save."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.core.hashes import avalanche_matrix, bucket_counts, clumping_hash, hash64
from src.evaluation.error_analysis import bloom_error_curve, bloom_memory_projection
from src.visualisation import (
    plot_avalanche_matrix,
    plot_bloom_error_curve,
    plot_bucket_histogram,
    plot_capacity_growth,
    plot_chain_lengths,
    plot_coin_flip_intuition,
    plot_count_min_overshoot,
    plot_feature_hashing_tradeoff,
    plot_hyperloglog_error,
    plot_lsh_s_curve,
    plot_memory_projection,
    plot_minhash_accuracy,
    plot_probe_cost,
    plot_table_occupancy,
)

VALUES = [f"user-{i}" for i in range(2_000)]


def test_hash_figures() -> None:
    good = bucket_counts(VALUES, 64, hash64)
    bad = bucket_counts(VALUES, 64, clumping_hash)
    assert isinstance(plot_bucket_histogram(good, bad), Figure)
    assert isinstance(plot_avalanche_matrix(avalanche_matrix(n_samples=50).matrix), Figure)


def test_hashtable_figures() -> None:
    frame = pd.DataFrame(
        {
            "strategy": ["chaining"] * 3 + ["open addressing"] * 3,
            "load_factor": [0.3, 0.6, 0.9] * 2,
            "mean_probes": [1.1, 1.3, 1.6, 1.2, 1.8, 5.5],
        }
    )
    assert isinstance(plot_probe_cost(frame), Figure)
    assert isinstance(plot_table_occupancy([0, 1, 1, 0, 1]), Figure)
    assert isinstance(plot_chain_lengths([0, 1, 1, 2, 3, 1, 0]), Figure)
    growth = pd.DataFrame({"n_items": [1, 10, 100], "capacity": [16, 16, 256]})
    assert isinstance(plot_capacity_growth(growth), Figure)


def test_bloom_figures(tmp_path: Path) -> None:
    curve = bloom_error_curve(expected_items=2_000, fp_targets=(0.1, 0.01), n_queries=2_000)
    figure = plot_bloom_error_curve(curve)
    assert isinstance(figure, Figure)
    assert isinstance(
        plot_memory_projection(bloom_memory_projection(scales=(10**6, 10**9))), Figure
    )

    from src.visualisation import save_figure

    saved = save_figure(figure, "test-bloom", directory=tmp_path)
    assert saved.exists() and saved.suffix == ".png"
    assert save_figure(figure, "test-bloom.png", directory=tmp_path) == saved


def test_coin_flip_figure() -> None:
    frame = pd.DataFrame({"n_values": [10, 100, 1_000], "longest_zero_run": [3, 7, 10]})
    assert isinstance(plot_coin_flip_intuition(frame), Figure)


def test_sketch_figures() -> None:
    hll = pd.DataFrame(
        {
            "precision": [10, 10, 12, 12],
            "true_cardinality": [1_000, 10_000, 1_000, 10_000],
            "relative_error": [0.02, -0.01, 0.01, 0.004],
            "standard_error": [0.032, 0.032, 0.016, 0.016],
            "packed_memory_bytes": [768, 768, 3_072, 3_072],
        }
    )
    assert isinstance(plot_hyperloglog_error(hll), Figure)
    count_min = pd.DataFrame({"true_count": [100, 200], "estimate": [105, 203]})
    assert isinstance(plot_count_min_overshoot(count_min), Figure)


def test_similarity_figures() -> None:
    minhash = pd.DataFrame(
        {
            "true_jaccard": [0.2, 0.8],
            "estimated_jaccard": [0.22, 0.78],
            "expected_std": [0.035, 0.035],
        }
    )
    assert isinstance(plot_minhash_accuracy(minhash), Figure)
    assert isinstance(plot_lsh_s_curve(((16, 8), (32, 4))), Figure)


def test_feature_hashing_figure() -> None:
    collisions = pd.DataFrame(
        {
            "n_buckets": [16, 256, 4096],
            "measured_collision_rate": [0.99, 0.6, 0.1],
            "expected_collision_rate": [0.99, 0.62, 0.11],
        }
    )
    model = pd.DataFrame({"n_buckets": [16, 256, 4096], "test_r2": [0.01, 0.3, 0.55]})
    assert isinstance(plot_feature_hashing_tradeoff(collisions), Figure)
    assert isinstance(plot_feature_hashing_tradeoff(collisions, model), Figure)


def test_figures_use_the_configured_dpi(tmp_path: Path) -> None:
    from src.config import get_settings
    from src.visualisation import save_figure

    figure = plot_lsh_s_curve()
    path = save_figure(figure, "s-curve", directory=tmp_path)
    assert path.stat().st_size > 1_000
    assert get_settings().figures.dpi > 0
    assert np.isfinite(figure.get_size_inches()).all()
