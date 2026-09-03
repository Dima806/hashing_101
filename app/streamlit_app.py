"""Four playgrounds for the four ideas: Bloom, HyperLogLog, near-duplicates, collisions.

Run with ``make run``. Everything here calls the same ``src/`` implementations the notebooks and
tests use - the app adds sliders, not algorithms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import get_settings
from src.core.hashtable import OpenAddressingHashTable
from src.data import generate_unique_items
from src.probabilistic.bloom import BloomFilter, sizing
from src.probabilistic.hyperloglog import HyperLogLog, standard_error
from src.similarity.lsh import MinHashLSH, approximate_threshold
from src.similarity.minhash import MinHasher, jaccard, shingles
from src.visualisation import plot_table_occupancy

st.set_page_config(page_title="Hashing 101", page_icon="#", layout="wide")

SETTINGS = get_settings()

DEFAULT_TEXTS = """The quick brown fox jumps over the lazy dog near the river bank.
The quick brown fox jumps over the lazy dog beside the river bank.
Probabilistic data structures trade a little accuracy for enormous savings in memory.
Probabilistic data structures trade a bit of accuracy for enormous savings in memory.
Completely unrelated text about the price of tea and the weather in November."""


@st.cache_resource(show_spinner=False)
def build_bloom(
    expected_items: int, target_fp_rate: float, n_added: int, seed: int
) -> BloomFilter:
    """A filter of the requested geometry, filled with ``n_added`` known items."""
    bloom = BloomFilter(expected_items, target_fp_rate, seed=seed)
    bloom.add_many([f"item-{i:09d}" for i in range(n_added)])
    return bloom


@st.cache_data(show_spinner=False)
def hyperloglog_trace(precision: int, n_unique: int, n_points: int, seed: int) -> pd.DataFrame:
    """Estimate against truth as a stream of distinct items arrives."""
    items = generate_unique_items(n_unique, seed=seed)
    sketch = HyperLogLog(precision=precision, seed=seed)
    checkpoints = np.unique(np.linspace(n_unique // n_points, n_unique, n_points).astype(int))
    rows = []
    seen = 0
    for checkpoint in checkpoints:
        sketch.add_many(items[seen:checkpoint])
        seen = int(checkpoint)
        estimate = sketch.estimate()
        rows.append(
            {
                "true_cardinality": seen,
                "estimate": estimate,
                "relative_error": (estimate - seen) / seen,
                "upper": seen * (1 + 2 * standard_error(sketch.m)),
                "lower": seen * (1 - 2 * standard_error(sketch.m)),
            }
        )
    return pd.DataFrame(rows)


def bloom_tab() -> None:
    """Add items, query others, watch the false positives appear as the filter fills."""
    st.subheader("Bloom filter playground")
    st.write(
        "A Bloom filter never says *no* about an item it has seen. It sometimes says *yes* about "
        "one it has not - at a rate you choose when you size it."
    )
    left, right = st.columns([1, 2])
    with left:
        expected_items = st.select_slider(
            "Sized for how many items?", options=[1_000, 10_000, 100_000, 1_000_000], value=10_000
        )
        target = st.select_slider(
            "Target false-positive rate", options=[0.2, 0.1, 0.05, 0.01, 0.001], value=0.01
        )
        fill = st.slider("Fill to this fraction of capacity", 0.1, 3.0, 1.0, 0.1)
        geometry = sizing(expected_items, target)
        st.metric("Memory", f"{geometry.memory_bytes / 1024:,.1f} KiB")
        st.metric("Bits per item", f"{geometry.bits_per_item:.1f}")
        st.metric("Hash functions", geometry.n_hashes)

    n_added = int(expected_items * fill)
    bloom = build_bloom(expected_items, target, n_added, SETTINGS.seed)
    absent = [f"absent-{i:09d}" for i in range(20_000)]
    measured_fp = float(bloom.contains_many(absent).mean())
    false_negatives = int((~bloom.contains_many([f"item-{i:09d}" for i in range(n_added)])).sum())

    with right:
        st.metric("Items added", f"{n_added:,}")
        columns = st.columns(3)
        columns[0].metric("Measured false-positive rate", f"{measured_fp:.3%}")
        columns[1].metric("Predicted by theory", f"{bloom.theoretical_fp_rate():.3%}")
        columns[2].metric(
            "False negatives", false_negatives, help="Always zero. That is the point."
        )
        st.progress(min(bloom.fill_ratio, 1.0), text=f"bits set: {bloom.fill_ratio:.1%}")
        if fill > 1.0:
            st.warning(
                "Past its design capacity the filter saturates: more bits are set, so more "
                "unseen items look present. The error grows - it never becomes a false negative."
            )
        probe = st.text_input("Query any string", value="item-000000042")
        if probe:
            verdict = probe in bloom
            st.write(
                f"**{probe}** -> `{verdict}`"
                + (
                    "  (definitely never added)"
                    if not verdict
                    else "  (probably added - or a false positive)"
                )
            )


def hyperloglog_tab() -> None:
    """Stream uniques and watch a kilobyte track a count in the millions."""
    st.subheader("HyperLogLog counter")
    st.write(
        "The estimate comes from the longest run of leading zeros in the hashes: improbable luck "
        "is evidence of many tries. The memory does not grow with the count."
    )
    left, right = st.columns([1, 2])
    with left:
        precision = st.slider(
            "Precision p (registers = 2^p)", 6, 16, SETTINGS.hyperloglog.precision
        )
        n_unique = st.select_slider(
            "Distinct items in the stream",
            options=[10_000, 50_000, 100_000, 500_000, 1_000_000],
            value=100_000,
        )
        sketch_size = HyperLogLog(precision=precision).packed_memory_bytes()
        st.metric("Sketch memory (packed)", f"{sketch_size:,} bytes")
        st.metric("Standard error", f"{standard_error(1 << precision):.2%}")
        st.metric("Exact set would need", f"~{n_unique * 100 / 1e6:,.1f} MB")

    trace = hyperloglog_trace(precision, n_unique, 12, SETTINGS.seed)
    with right:
        st.line_chart(
            trace.set_index("true_cardinality")[["estimate", "upper", "lower"]],
            height=320,
        )
        final = trace.iloc[-1]
        columns = st.columns(3)
        columns[0].metric("True cardinality", f"{int(final['true_cardinality']):,}")
        columns[1].metric("Estimate", f"{final['estimate']:,.0f}")
        columns[2].metric("Relative error", f"{final['relative_error']:+.2%}")


def near_duplicate_tab() -> None:
    """Paste texts, watch MinHash and LSH group the near-identical ones."""
    st.subheader("Near-duplicate finder")
    st.write(
        "One line per document. MinHash turns each into a signature; LSH buckets the similar."
    )
    texts = st.text_area("Documents", value=DEFAULT_TEXTS, height=160)
    documents = [line.strip() for line in texts.splitlines() if line.strip()]
    left, right = st.columns(2)
    with left:
        shingle_size = st.slider("Shingle size (words)", 1, 8, 3)
        num_bands = st.select_slider("Bands", options=[8, 16, 32, 64], value=32)
    with right:
        rows_per_band = st.select_slider("Rows per band", options=[2, 4, 8, 16], value=4)
        st.metric("Similarity threshold", f"{approximate_threshold(num_bands, rows_per_band):.2f}")

    if len(documents) < 2:
        st.info("Add at least two documents.")
        return

    hasher = MinHasher(num_perm=num_bands * rows_per_band, seed=SETTINGS.seed)
    index = MinHashLSH(num_bands=num_bands, rows_per_band=rows_per_band, seed=SETTINGS.seed)
    shingle_sets = [shingles(document, shingle_size) for document in documents]
    for i, shingle_set in enumerate(shingle_sets):
        index.insert(i, hasher.signature(shingle_set))

    pairs = sorted(index.candidate_pairs())
    if not pairs:
        st.info("No candidate pairs at this threshold - lower it, or shorten the shingles.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "document A": documents[int(a)][:60],
                    "document B": documents[int(b)][:60],
                    "estimated Jaccard": hasher.estimated_jaccard(
                        hasher.signature(shingle_sets[int(a)]),
                        hasher.signature(shingle_sets[int(b)]),
                    ),
                    "true Jaccard": jaccard(shingle_sets[int(a)], shingle_sets[int(b)]),
                }
                for a, b in pairs
            ]
        ),
        width="stretch",
    )
    st.caption(
        f"{len(pairs)} candidate pairs out of {len(documents) * (len(documents) - 1) // 2} "
        "possible - that ratio is the whole reason LSH exists."
    )


def collision_tab() -> None:
    """Hash items into a small table and watch collisions pile up with the load factor."""
    st.subheader("Collision visualiser")
    st.write(
        "Every slot is a bucket. As the table fills, probes get longer - flat, then a wall. "
        "That wall is why a dict resizes itself."
    )
    capacity = st.select_slider("Table slots", options=[32, 64, 128, 256, 512], value=128)
    load_factor = st.slider("Load factor", 0.05, 0.99, 0.6, 0.01)
    n_items = max(1, int(capacity * load_factor))

    table = OpenAddressingHashTable(capacity=capacity, auto_resize=False, seed=SETTINGS.seed)
    for i in range(n_items):
        table[f"key-{i:05d}"] = i
    table.reset_probe_stats()
    for i in range(n_items):
        _ = table[f"key-{i:05d}"]
    stats = table.probe_stats()

    columns = st.columns(3)
    columns[0].metric("Items", n_items)
    columns[1].metric("Load factor", f"{stats.load_factor:.2f}")
    columns[2].metric("Mean probes per lookup", f"{stats.mean_probes:.2f}")
    st.pyplot(plot_table_occupancy(table.occupancy()))
    st.caption(
        "Linear probing: expected probes for a successful lookup are about "
        "(1 + 1/(1-a))/2, which is 1.5 at a=0.5 and 5.5 at a=0.9."
    )


def main() -> None:
    """Render the app."""
    st.title("Hashing 101")
    st.caption(
        "How to remember a billion things in a megabyte and be wrong on purpose. "
        "Every structure here is built from scratch in `src/`."
    )
    bloom, hll, duplicates, collisions = st.tabs(
        ["Bloom filter", "HyperLogLog", "Near-duplicates", "Collisions"]
    )
    with bloom:
        bloom_tab()
    with hll:
        hyperloglog_tab()
    with duplicates:
        near_duplicate_tab()
    with collisions:
        collision_tab()


main()
