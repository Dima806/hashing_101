"""Every figure in the project, so notebooks stay narrative and the plots stay consistent.

Each function takes a DataFrame produced by ``src/evaluation/`` and returns a matplotlib Figure.
Nothing here computes anything the notebooks could not; it just draws it the same way twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.config import get_settings
from src.similarity.lsh import candidate_probability

ACCENT = "#2f6f9f"
CONTRAST = "#c1462f"
MUTED = "#8a8f98"


def _finish(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(figure: Figure, name: str, directory: Path | None = None) -> Path:
    """Write a figure to ``outputs/figures`` (created if needed) and return its path."""
    settings = get_settings()
    target_dir = directory if directory is not None else settings.figures.path
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / (name if name.endswith(".png") else f"{name}.png")
    figure.savefig(path, dpi=settings.figures.dpi, bbox_inches="tight")
    return path


def plot_bucket_histogram(
    good_counts: np.ndarray, bad_counts: np.ndarray, figsize: tuple[float, float] = (10.0, 3.6)
) -> Figure:
    """Two hashes, the same inputs: one spreads them, one piles them up (notebook 01)."""
    figure = Figure(figsize=figsize)
    axes = figure.subplots(1, 2, sharey=True)
    for ax, counts, title, colour in (
        (axes[0], good_counts, "hash64: uniform", ACCENT),
        (axes[1], bad_counts, "sum of bytes: clumped", CONTRAST),
    ):
        ax.bar(np.arange(len(counts)), counts, color=colour, width=1.0)
        _finish(ax, title, "bucket", "items")
    figure.tight_layout()
    return figure


def plot_avalanche_matrix(matrix: np.ndarray, figsize: tuple[float, float] = (5.0, 4.4)) -> Figure:
    """Flip one input bit, see which output bits move. A good hash is flat 0.5 everywhere."""
    figure = Figure(figsize=figsize)
    ax = figure.subplots()
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdBu", aspect="auto")
    figure.colorbar(image, ax=ax, label="P(output bit flips)")
    _finish(ax, "Avalanche", "output bit", "flipped input bit")
    figure.tight_layout()
    return figure


def plot_probe_cost(frame: pd.DataFrame, figsize: tuple[float, float] = (6.4, 4.0)) -> Figure:
    """Mean probes against load factor: flat, then the wall (notebook 02)."""
    figure = Figure(figsize=figsize)
    ax = figure.subplots()
    for strategy, group in frame.groupby("strategy"):
        colour = ACCENT if "chain" in str(strategy).lower() else CONTRAST
        ax.plot(
            group["load_factor"],
            group["mean_probes"],
            marker="o",
            color=colour,
            label=str(strategy),
        )
        if "theory_probes" in group.columns:
            ax.plot(
                group["load_factor"],
                group["theory_probes"],
                linestyle="--",
                color=colour,
                alpha=0.6,
                label=f"{strategy} (theory)",
            )
    ax.legend(frameon=False, fontsize=8)
    _finish(ax, "Lookup cost against load factor", "load factor", "mean probes per lookup")
    figure.tight_layout()
    return figure


def plot_chain_lengths(
    lengths: Sequence[int], figsize: tuple[float, float] = (6.2, 3.6)
) -> Figure:
    """How many slots hold 0, 1, 2, ... items - the Poisson shape a good hash produces."""
    figure = Figure(figsize=figsize)
    ax = figure.subplots()
    values, counts = np.unique(np.asarray(lengths), return_counts=True)
    ax.bar(values, counts, color=ACCENT, width=0.8)
    _finish(ax, "Chain lengths", "items in a slot", "slots")
    figure.tight_layout()
    return figure


def plot_capacity_growth(frame: pd.DataFrame, figsize: tuple[float, float] = (6.4, 4.0)) -> Figure:
    """Capacity doubling against items inserted: the sawtooth that keeps lookups flat."""
    figure = Figure(figsize=figsize)
    ax = figure.subplots()
    ax.step(frame["n_items"], frame["capacity"], where="post", color=ACCENT, label="capacity")
    ax.plot(frame["n_items"], frame["n_items"], color=MUTED, linestyle="--", label="items")
    ax.legend(frameon=False)
    _finish(ax, "A dictionary resizing itself", "items inserted", "slots allocated")
    figure.tight_layout()
    return figure


def plot_coin_flip_intuition(
    frame: pd.DataFrame, figsize: tuple[float, float] = (6.4, 4.2)
) -> Figure:
    """Longest run of leading zeros against how many values were seen - HyperLogLog in one plot."""
    figure = Figure(figsize=figsize)
    ax = figure.subplots()
    ax.plot(
        frame["n_values"],
        frame["longest_zero_run"],
        marker="o",
        color=ACCENT,
        label="longest run seen",
    )
    ax.plot(
        frame["n_values"], np.log2(frame["n_values"]), color=MUTED, linestyle="--", label="log2(n)"
    )
    ax.set_xscale("log")
    ax.legend(frameon=False)
    _finish(
        ax, "Improbable luck counts the tries", "values hashed", "longest run of leading zeros"
    )
    figure.tight_layout()
    return figure


def plot_bloom_error_curve(
    frame: pd.DataFrame, figsize: tuple[float, float] = (6.4, 4.2)
) -> Figure:
    """Memory against false-positive rate - the figure that sells the whole idea."""
    figure = Figure(figsize=figsize)
    ax = figure.subplots()
    ax.plot(
        frame["memory_bytes"] / 1024,
        frame["theoretical_fp_rate"],
        color=MUTED,
        linestyle="--",
        label="theory",
    )
    ax.scatter(
        frame["memory_bytes"] / 1024,
        frame["measured_fp_rate"],
        color=ACCENT,
        zorder=3,
        label="measured",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    _finish(ax, "Bloom filter: memory buys accuracy", "memory (KiB)", "false-positive rate")
    figure.tight_layout()
    return figure


def plot_memory_projection(
    frame: pd.DataFrame, figsize: tuple[float, float] = (6.4, 4.2)
) -> Figure:
    """Exact set against Bloom filter as the item count grows to a billion (projected)."""
    figure = Figure(figsize=figsize)
    ax = figure.subplots()
    ax.plot(
        frame["n_items"],
        frame["exact_set_bytes"] / 1e9,
        marker="o",
        color=CONTRAST,
        label="exact set (projected)",
    )
    ax.plot(
        frame["n_items"],
        frame["bloom_bytes"] / 1e9,
        marker="o",
        color=ACCENT,
        label="bloom filter",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    _finish(ax, "Memory to track n items", "items", "gigabytes")
    figure.tight_layout()
    return figure


def plot_hyperloglog_error(
    frame: pd.DataFrame, figsize: tuple[float, float] = (6.6, 4.2)
) -> Figure:
    """Relative error against true cardinality, with the theoretical band per precision."""
    figure = Figure(figsize=figsize)
    ax = figure.subplots()
    palette = [ACCENT, CONTRAST, "#5a8f3d", "#8a5fb0", MUTED]
    for colour, (precision, group) in zip(palette, frame.groupby("precision"), strict=False):
        ax.plot(
            group["true_cardinality"],
            100 * group["relative_error"],
            marker="o",
            color=colour,
            label=f"p={precision} ({group['packed_memory_bytes'].iloc[0]} B)",
        )
        band = 100 * group["standard_error"].iloc[0]
        ax.axhspan(-band, band, color=colour, alpha=0.06)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8)
    _finish(
        ax,
        "HyperLogLog error is fixed, whatever the count",
        "true cardinality",
        "relative error (%)",
    )
    figure.tight_layout()
    return figure


def plot_count_min_overshoot(
    frame: pd.DataFrame, figsize: tuple[float, float] = (6.4, 4.0)
) -> Figure:
    """True counts against estimates for the heavy hitters: on or above the diagonal, never below."""
    figure = Figure(figsize=figsize)
    ax = figure.subplots()
    limit = float(max(frame["estimate"].max(), frame["true_count"].max())) * 1.05
    ax.plot([0, limit], [0, limit], color=MUTED, linestyle="--", label="exact")
    ax.scatter(frame["true_count"], frame["estimate"], color=ACCENT, zorder=3, label="estimate")
    ax.legend(frameon=False)
    _finish(ax, "Count-Min never underestimates", "true count", "estimated count")
    figure.tight_layout()
    return figure


def plot_minhash_accuracy(
    frame: pd.DataFrame, figsize: tuple[float, float] = (6.0, 4.2)
) -> Figure:
    """Estimated Jaccard against the truth, with the sampling band the signature length allows."""
    figure = Figure(figsize=figsize)
    ax = figure.subplots()
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", label="exact")
    ax.errorbar(
        frame["true_jaccard"],
        frame["estimated_jaccard"],
        yerr=3 * frame["expected_std"],
        fmt="o",
        color=ACCENT,
        capsize=3,
        label="MinHash estimate (3 sigma)",
    )
    ax.legend(frameon=False)
    _finish(ax, "Signature agreement is Jaccard similarity", "true Jaccard", "estimated Jaccard")
    figure.tight_layout()
    return figure


def plot_lsh_s_curve(
    bandings: Sequence[tuple[int, int]] = ((8, 16), (16, 8), (32, 4)),
    figsize: tuple[float, float] = (6.2, 4.2),
) -> Figure:
    """The S-curve: which similarities a banding will even consider (notebook 05)."""
    figure = Figure(figsize=figsize)
    ax = figure.subplots()
    similarities = np.linspace(0.0, 1.0, 201)
    palette = [ACCENT, CONTRAST, "#5a8f3d", "#8a5fb0"]
    for colour, (bands, rows) in zip(palette, bandings, strict=False):
        probabilities = [candidate_probability(float(s), bands, rows) for s in similarities]
        ax.plot(similarities, probabilities, color=colour, label=f"b={bands}, r={rows}")
    ax.legend(frameon=False)
    _finish(ax, "LSH candidate probability", "true similarity", "P(becomes a candidate)")
    figure.tight_layout()
    return figure


def plot_feature_hashing_tradeoff(
    collision_frame: pd.DataFrame,
    model_frame: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (6.6, 4.2),
) -> Figure:
    """Collisions falling and model quality rising as buckets grow - the tradeoff, drawn."""
    figure = Figure(figsize=figsize)
    ax = figure.subplots()
    ax.plot(
        collision_frame["n_buckets"],
        collision_frame["measured_collision_rate"],
        marker="o",
        color=CONTRAST,
        label="collision rate (measured)",
    )
    ax.plot(
        collision_frame["n_buckets"],
        collision_frame["expected_collision_rate"],
        linestyle="--",
        color=MUTED,
        label="collision rate (theory)",
    )
    ax.set_xscale("log", base=2)
    _finish(ax, "Feature hashing: buckets against collisions", "buckets", "collision rate")
    if model_frame is not None:
        twin = ax.twinx()
        twin.plot(
            model_frame["n_buckets"],
            model_frame["test_r2"],
            marker="s",
            color=ACCENT,
            label="model R^2",
        )
        twin.set_ylabel("test R^2")
        twin.spines["top"].set_visible(False)
        handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
        labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
        ax.legend(handles, labels, frameon=False, fontsize=8, loc="center right")
    else:
        ax.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return figure


def plot_table_occupancy(
    occupancy: Sequence[int], figsize: tuple[float, float] = (7.0, 1.6)
) -> Figure:
    """A hash table's slots as a strip of filled and empty cells (the collision visualiser)."""
    figure = Figure(figsize=figsize)
    ax = figure.subplots()
    ax.imshow(np.asarray(occupancy).reshape(1, -1), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_xlabel("slot")
    ax.set_title(f"occupancy: {sum(occupancy)}/{len(occupancy)} slots used")
    figure.tight_layout()
    return figure
