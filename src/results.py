"""Persist every number and every sentence a notebook produces, as JSON.

Notebooks are for reading; ``outputs/results/*.json`` is for reusing. Every table, measurement and
takeaway a notebook computes is written here, so the README and any later analysis quote the same
numbers the code actually produced - and a reviewer can diff them after a rerun without opening a
notebook.

Output carries no timestamps, so a rerun of the same seed rewrites the same values - with one
honest exception: files that record wall-clock timings (``*_exact_vs_approximate``,
``02_probe_cost``, ``01_backends``) differ run to run by however much the machine differed. Every
*measured statistic* is stable; only the seconds move.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_settings


def results_dir() -> Path:
    """The directory JSON results are written to (created on demand)."""
    path = get_settings().results.path
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean(value: Any) -> Any:
    """Convert numpy, pandas and dataclass values into something ``json`` can write.

    Non-finite floats become ``null`` rather than the ``NaN`` literal, which is not valid JSON and
    breaks every strict parser downstream.
    """
    if isinstance(value, pd.DataFrame):
        return [clean(record) for record in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [clean(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return clean(asdict(value))
    if value is None or isinstance(value, str):
        return value
    return repr(value)


def save_results(payload: Mapping[str, Any], name: str, directory: Path | None = None) -> Path:
    """Write ``payload`` to ``outputs/results/<name>.json`` and return the path.

    ``payload`` can hold DataFrames, numpy scalars, dataclasses, sets and plain text - anything a
    notebook naturally ends up with.
    """
    target_dir = directory if directory is not None else results_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / (name if name.endswith(".json") else f"{name}.json")
    document = {"_meta": {"name": path.stem, "seed": get_settings().seed}, **clean(payload)}
    path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def load_results(name: str, directory: Path | None = None) -> dict[str, Any]:
    """Read back a JSON result file written by :func:`save_results`."""
    target_dir = directory if directory is not None else results_dir()
    path = target_dir / (name if name.endswith(".json") else f"{name}.json")
    return json.loads(path.read_text(encoding="utf-8"))
