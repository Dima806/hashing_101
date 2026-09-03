"""The notebooks and their artefacts are deliverables, so they are checked like everything else.

These tests do not execute the notebooks (``make notebooks`` does that, in about a minute). They
check the committed result: six notebooks, executed, error-free, and every figure and every number
they claim to save actually present on disk as a PNG or a strict-JSON file.
"""

from __future__ import annotations

import json
import re

import nbformat
import pytest

from src.config import PROJECT_ROOT, get_settings

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
EXPECTED = [
    "01_what_a_hash_really_is.ipynb",
    "02_hash_tables_from_scratch.ipynb",
    "03_bloom_filters.ipynb",
    "04_counting_unique_with_hyperloglog.ipynb",
    "05_near_duplicates_minhash_lsh.ipynb",
    "06_feature_hashing_and_guide.ipynb",
]

_FIGURE_CALL = re.compile(r"save_figure\([^,]+,\s*[\"']([\w.-]+)[\"']")
_RESULT_CALL = re.compile(r"save_results\(.*?[\"']([\w.-]+)[\"'],?\s*\)", re.S)


def _notebook(name: str) -> nbformat.NotebookNode:
    return nbformat.read(NOTEBOOK_DIR / name, as_version=4)


def _code(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")


def _require_artefacts(directory) -> None:
    if not directory.exists() or not any(directory.iterdir()):
        pytest.skip(f"{directory.name} is empty - run `make notebooks` to regenerate artefacts")


@pytest.mark.parametrize("name", EXPECTED)
def test_notebook_exists(name: str) -> None:
    assert (NOTEBOOK_DIR / name).exists(), f"{name} is missing"


@pytest.mark.parametrize("name", EXPECTED)
def test_notebook_ran_without_errors(name: str) -> None:
    """A committed notebook carries its outputs, and none of them may be a traceback."""
    notebook = _notebook(name)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    assert not errors, f"{name} has {len(errors)} error output(s): {errors[0].get('ename')}"
    assert any(cell.get("outputs") for cell in code_cells), f"{name} was never executed"


@pytest.mark.parametrize("name", EXPECTED)
def test_notebook_has_narrative(name: str) -> None:
    """These are teaching notebooks: prose is the deliverable, code is the evidence."""
    notebook = _notebook(name)
    markdown = [cell for cell in notebook.cells if cell.cell_type == "markdown"]
    assert len(markdown) >= 4, f"{name} has only {len(markdown)} markdown cells"
    assert markdown[0].source.startswith("# "), f"{name} does not open with a title"


@pytest.mark.parametrize("name", EXPECTED)
def test_every_figure_is_saved_as_png(name: str) -> None:
    figures_dir = get_settings().figures.path
    _require_artefacts(figures_dir)
    declared = set(_FIGURE_CALL.findall(_code(_notebook(name))))
    assert declared, f"{name} saves no figures"
    for figure_name in declared:
        path = figures_dir / (
            figure_name if figure_name.endswith(".png") else f"{figure_name}.png"
        )
        assert path.exists(), f"{name} saves {figure_name} but {path.name} is missing"
        assert path.stat().st_size > 1_000


@pytest.mark.parametrize("name", EXPECTED)
def test_every_result_is_saved_as_strict_json(name: str) -> None:
    results_path = get_settings().results.path
    _require_artefacts(results_path)
    declared = set(_RESULT_CALL.findall(_code(_notebook(name))))
    assert declared, f"{name} saves no results"
    for result_name in declared:
        path = results_path / f"{result_name}.json"
        assert path.exists(), f"{name} saves {result_name} but {path.name} is missing"
        text = path.read_text()
        assert "NaN" not in text and "Infinity" not in text
        document = json.loads(text)
        assert document["_meta"]["name"] == result_name
        assert len(document) > 1, f"{path.name} holds only metadata"


def test_no_orphan_artefacts() -> None:
    """Every PNG and JSON in outputs/ is claimed by a notebook - nothing stale left behind."""
    settings = get_settings()
    _require_artefacts(settings.figures.path)
    sources = {name: _code(_notebook(name)) for name in EXPECTED}
    claimed_figures = {
        figure for source in sources.values() for figure in _FIGURE_CALL.findall(source)
    }
    claimed_results = {
        result for source in sources.values() for result in _RESULT_CALL.findall(source)
    }
    on_disk_figures = {path.stem for path in settings.figures.path.glob("*.png")}
    on_disk_results = {path.stem for path in settings.results.path.glob("*.json")}
    assert on_disk_figures - claimed_figures == set()
    assert on_disk_results - claimed_results == set()


def test_headline_numbers_survive_in_json() -> None:
    """The claims the README quote must be readable straight out of outputs/."""
    results_path = get_settings().results.path
    _require_artefacts(results_path)
    bloom = json.loads((results_path / "03_bloom_guarantee.json").read_text())
    assert bloom["false_negatives"] == 0
    assert bloom["measured_fp_rate"] < 0.02

    projection = json.loads((results_path / "03_billion_item_projection.json").read_text())
    assert projection["billion_items_bloom_gb"] == pytest.approx(1.2, abs=0.1)
    assert projection["savings_factor"] > 50

    hyperloglog = json.loads((results_path / "04_hyperloglog_headline.json").read_text())
    assert hyperloglog["packed_memory_bytes"] == 1_536
    assert abs(hyperloglog["relative_error"]) < 4 * hyperloglog["standard_error"]

    duplicates = json.loads((results_path / "05_found_duplicates.json").read_text())
    assert duplicates["planted_recovered"] >= 0.85 * duplicates["planted_pairs"]
    assert duplicates["candidate_pairs"] < duplicates["all_pairs"] / 50
