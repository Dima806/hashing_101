"""The from-scratch dictionary must behave exactly like the built-in one, and cost what theory says."""

from __future__ import annotations

import random

import pytest

from src.core.hashtable import ChainingHashTable, OpenAddressingHashTable

TABLES = [ChainingHashTable, OpenAddressingHashTable]


@pytest.mark.parametrize("table_class", TABLES)
def test_matches_a_real_dict_under_random_operations(table_class: type) -> None:
    rng = random.Random(11)
    table = table_class(capacity=8, seed=3)
    oracle: dict[str, int] = {}
    for step in range(3_000):
        key = f"key-{rng.randrange(400)}"
        if rng.random() < 0.7:
            table[key] = step
            oracle[key] = step
        elif key in oracle:
            del table[key]
            del oracle[key]
        assert (key in table) == (key in oracle)
    assert len(table) == len(oracle)
    assert sorted(table) == sorted(oracle)
    for key, value in oracle.items():
        assert table[key] == value


@pytest.mark.parametrize("table_class", TABLES)
def test_missing_key_raises(table_class: type) -> None:
    table = table_class(capacity=8)
    with pytest.raises(KeyError):
        _ = table["nothing"]
    with pytest.raises(KeyError):
        del table["nothing"]
    assert table.get("nothing", "fallback") == "fallback"


@pytest.mark.parametrize("table_class", TABLES)
def test_resizing_preserves_contents_and_bounds_the_load_factor(table_class: type) -> None:
    table = table_class(capacity=8, max_load_factor=0.75)
    for i in range(2_000):
        table[f"key-{i}"] = i
    assert table.n_resizes > 0
    assert table.load_factor <= 0.75
    assert all(table[f"key-{i}"] == i for i in range(0, 2_000, 97))


def test_open_addressing_survives_deletion_in_a_probe_chain() -> None:
    """Deleting must leave a tombstone, or keys behind the hole become unreachable."""
    table = OpenAddressingHashTable(capacity=16, auto_resize=False, seed=1)
    keys = [f"k{i}" for i in range(10)]
    for i, key in enumerate(keys):
        table[key] = i
    del table[keys[3]]
    assert keys[3] not in table
    for i, key in enumerate(keys):
        if i != 3:
            assert table[key] == i


def test_open_addressing_reuses_tombstones() -> None:
    table = OpenAddressingHashTable(capacity=16, auto_resize=False, seed=1)
    for i in range(8):
        table[f"k{i}"] = i
    del table["k4"]
    assert table.n_tombstones == 1
    table["k4"] = 44
    assert table["k4"] == 44
    assert table.n_tombstones == 0


def test_open_addressing_refuses_to_overfill_without_resizing() -> None:
    table = OpenAddressingHashTable(capacity=4, max_load_factor=1.0, auto_resize=False)
    for i in range(4):
        table[f"k{i}"] = i
    with pytest.raises(RuntimeError, match="full"):
        table["one-too-many"] = 5


def test_probe_cost_explodes_as_the_table_fills() -> None:
    """Flat, then a wall - the figure notebook 02 plots, asserted as a fact."""
    costs = {}
    for load_factor in (0.3, 0.9):
        capacity = 4_096
        table = OpenAddressingHashTable(capacity=capacity, auto_resize=False, seed=2)
        n_items = int(capacity * load_factor)
        for i in range(n_items):
            table[f"key-{i}"] = i
        table.reset_probe_stats()
        for i in range(n_items):
            _ = table[f"key-{i}"]
        costs[load_factor] = table.probe_stats().mean_probes
    assert costs[0.3] < 1.5
    assert costs[0.9] > 3 * costs[0.3]


@pytest.mark.parametrize("table_class", TABLES)
def test_memory_and_probe_bookkeeping(table_class: type) -> None:
    table = table_class(capacity=64)
    for i in range(32):
        table[f"key-{i}"] = i
    _ = table["key-0"]
    stats = table.probe_stats()
    assert stats.lookups == 1
    assert stats.probes >= 1
    assert stats.load_factor == pytest.approx(0.5)
    assert table.memory_bytes() > 0
    table.reset_probe_stats()
    assert table.probe_stats().lookups == 0


@pytest.mark.parametrize("table_class", TABLES)
def test_invalid_geometry_is_rejected(table_class: type) -> None:
    with pytest.raises(ValueError, match="capacity"):
        table_class(capacity=0)
    with pytest.raises(ValueError, match="max_load_factor"):
        table_class(capacity=8, max_load_factor=1.5)
