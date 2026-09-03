"""Generated data is only useful if its ground truth is exact. These tests are that guarantee."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import (
    Corpus,
    Stream,
    generate_categorical_table,
    generate_stream,
    generate_text_corpus,
    generate_unique_items,
)
from src.similarity.minhash import jaccard, shingles


def test_stream_cardinality_is_exact(stream: Stream) -> None:
    assert stream.true_cardinality == 5_000
    assert stream.n_items == 50_000
    assert len(set(stream.items)) == 5_000


def test_stream_is_reproducible() -> None:
    first = generate_stream(1_000, 100, seed=3)
    assert first.items == generate_stream(1_000, 100, seed=3).items
    assert first.items != generate_stream(1_000, 100, seed=4).items


def test_zipf_stream_has_heavy_hitters(stream: Stream) -> None:
    top_count = stream.top_k(1)[0][1]
    assert top_count > 20 * (stream.n_items / stream.true_cardinality)


def test_uniform_stream_is_flat() -> None:
    uniform = generate_stream(50_000, 1_000, seed=1, distribution="uniform")
    counts = np.array([count for _, count in uniform.true_counts.items()])
    assert counts.std() / counts.mean() < 0.2


def test_absent_items_are_absent(stream: Stream) -> None:
    assert not set(stream.absent_items(100)) & set(stream.items)


def test_unique_items_are_unique() -> None:
    items = generate_unique_items(1_000, seed=1)
    assert len(set(items)) == 1_000


def test_planted_duplicates_are_actually_near_duplicates(corpus: Corpus) -> None:
    """LSH recall is only meaningful if the ground truth pairs really are similar."""
    shingle_sets = [shingles(document, 5) for document in corpus.documents]
    similarities = [jaccard(shingle_sets[a], shingle_sets[b]) for a, b in corpus.duplicate_pairs]
    assert min(similarities) > 0.5
    assert len(corpus.duplicate_pairs) == 15
    assert corpus.n_documents == 120


def test_unrelated_documents_are_not_similar(corpus: Corpus) -> None:
    shingle_sets = [shingles(document, 5) for document in corpus.documents]
    planted = {(min(a, b), max(a, b)) for a, b in corpus.duplicate_pairs}
    unrelated = [
        jaccard(shingle_sets[i], shingle_sets[j])
        for i in range(0, 40)
        for j in range(i + 1, 40)
        if (i, j) not in planted
    ]
    assert max(unrelated) < 0.2


def test_corpus_rejects_impossible_requests() -> None:
    with pytest.raises(ValueError, match="n_near_duplicates"):
        generate_text_corpus(n_docs=10, n_near_duplicates=20)


def test_stream_rejects_impossible_cardinality() -> None:
    with pytest.raises(ValueError, match="n_items"):
        generate_stream(n_items=10, n_unique=100)
    with pytest.raises(ValueError, match="distribution"):
        generate_stream(100, 10, distribution="pareto")


def test_categorical_table_has_signal_in_the_high_cardinality_column() -> None:
    table = generate_categorical_table(n_rows=2_000, n_categories=200, n_informative=50, seed=1)
    assert set(table.columns) == {"user_id", "city", "device", "y"}
    assert table["user_id"].nunique() > 150
    group_means = table.groupby("user_id")["y"].mean()
    assert group_means.std() > 0.3  # the target really does depend on the category
