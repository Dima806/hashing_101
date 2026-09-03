"""Settings are the single place scale is chosen, so the file itself has to stay coherent."""

from __future__ import annotations

from src.config import CONFIG_PATH, PROJECT_ROOT, Settings, get_settings


def test_settings_load_from_yaml() -> None:
    assert CONFIG_PATH.exists()
    settings = get_settings()
    assert settings.seed > 0
    assert settings.hyperloglog.precision >= 4
    assert 0 < settings.bloom.target_fp_rate < 1


def test_settings_are_cached() -> None:
    assert get_settings() is get_settings()


def test_minhash_and_lsh_agree_on_signature_length() -> None:
    """A signature has to divide exactly into bands - a mismatch would fail at insert time."""
    settings = get_settings()
    assert settings.lsh.num_bands * settings.lsh.rows_per_band == settings.minhash.num_perm


def test_environment_overrides_yaml(monkeypatch) -> None:
    monkeypatch.setenv("HASHING101_SEED", "12345")
    assert Settings().seed == 12345


def test_figure_directory_resolves_under_the_project() -> None:
    settings = get_settings()
    assert settings.figures.path.is_relative_to(PROJECT_ROOT)
    assert settings.figures.dpi > 0
