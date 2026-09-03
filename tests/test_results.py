"""Notebook results are a deliverable: valid, strict JSON with nothing numpy-shaped left in it."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_settings
from src.results import clean, load_results, results_dir, save_results


@dataclass(frozen=True)
class _Sizing:
    n_bits: int
    label: str


def test_round_trip(tmp_path: Path) -> None:
    payload = {
        "table": pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]}),
        "counts": np.array([1, 2, 3]),
        "scalar": np.float64(0.25),
        "flag": np.bool_(True),
        "sizing": _Sizing(n_bits=10, label="tiny"),
        "takeaway": "a bad hash quietly destroys everything built on it",
        "pairs": {(1, 2)},
    }
    path = save_results(payload, "unit-test", directory=tmp_path)
    loaded = load_results("unit-test", directory=tmp_path)
    assert path.name == "unit-test.json"
    assert loaded["table"] == [{"a": 1, "b": 0.5}, {"a": 2, "b": 1.5}]
    assert loaded["counts"] == [1, 2, 3]
    assert loaded["scalar"] == 0.25
    assert loaded["flag"] is True
    assert loaded["sizing"] == {"n_bits": 10, "label": "tiny"}
    assert loaded["_meta"]["seed"] == get_settings().seed


def test_output_is_strict_json_without_nan(tmp_path: Path) -> None:
    """``NaN`` is not valid JSON - every strict parser downstream would choke on it."""
    path = save_results({"value": float("nan"), "big": math.inf}, "nan-test", directory=tmp_path)
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    parsed = json.loads(text, parse_constant=_reject_constant)
    assert parsed["value"] is None
    assert parsed["big"] is None


def _reject_constant(value: str) -> float:
    raise AssertionError(f"JSON contained the non-finite constant {value!r}")


def test_clean_handles_nesting() -> None:
    cleaned = clean({"rows": [{"x": np.int64(1)}, (np.float32(2.5), "text")]})
    assert cleaned == {"rows": [{"x": 1}, [2.5, "text"]]}


def test_is_deterministic(tmp_path: Path) -> None:
    """The same seed must produce byte-identical files, so a diff means a result changed."""
    payload = {"numbers": [1, 2, 3], "note": "stable"}
    first = save_results(payload, "stable", directory=tmp_path).read_bytes()
    second = save_results(payload, "stable", directory=tmp_path).read_bytes()
    assert first == second


def test_results_dir_is_under_the_project() -> None:
    assert results_dir().is_relative_to(get_settings().results.path.parent)
