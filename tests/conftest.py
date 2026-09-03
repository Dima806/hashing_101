"""Shared fixtures. Everything is seeded; nothing here downloads or writes outside tmp_path."""

from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.data import (  # noqa: E402  (import after backend selection is deliberate)
    Corpus,
    Stream,
    generate_categorical_table,
    generate_stream,
    generate_text_corpus,
)

SEED = 7


@pytest.fixture(scope="session")
def stream() -> Stream:
    """A Zipf-distributed stream with exactly 5,000 distinct items."""
    return generate_stream(n_items=50_000, n_unique=5_000, seed=SEED)


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    """120 documents with 15 planted near-duplicate pairs."""
    return generate_text_corpus(
        n_docs=120, n_near_duplicates=15, doc_words=60, vocabulary_size=300, seed=SEED
    )


@pytest.fixture(scope="session")
def categorical_table() -> pd.DataFrame:
    """A small high-cardinality table with a learnable target."""
    return generate_categorical_table(
        n_rows=3_000, n_categories=800, n_informative=120, noise=0.3, seed=SEED
    )
