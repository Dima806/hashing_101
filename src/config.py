"""Typed access to ``config/settings.yaml``.

Nothing in ``src/``, the notebooks or the app should hard-code a bucket count, a hash count, an
error target or a seed: they all come from here, so a reader can dial the whole project up or
down (notebook runtime, memory) by editing one file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class StreamSettings(BaseModel):
    """Synthetic item stream with a known unique count and known frequencies."""

    n_items: int = 200_000
    n_unique: int = 20_000
    distribution: str = "zipf"
    zipf_exponent: float = 1.2


class HashTableSettings(BaseModel):
    initial_capacity: int = 16
    max_load_factor: float = 0.75
    demo_items: int = 20_000


class BloomSettings(BaseModel):
    expected_items: int = 100_000
    target_fp_rate: float = 0.01
    fp_targets: list[float] = Field(default_factory=lambda: [0.1, 0.05, 0.01, 0.001, 0.0001])
    projection_scales: list[int] = Field(
        default_factory=lambda: [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
    )


class CountingBloomSettings(BaseModel):
    expected_items: int = 50_000
    target_fp_rate: float = 0.01
    counter_bits: int = 8


class HyperLogLogSettings(BaseModel):
    precision: int = 11
    precisions: list[int] = Field(default_factory=lambda: [8, 10, 12, 14])
    cardinalities: list[int] = Field(default_factory=lambda: [1_000, 10_000, 100_000, 1_000_000])


class CountMinSettings(BaseModel):
    epsilon: float = 0.001
    delta: float = 0.01
    top_k: int = 10


class MinHashSettings(BaseModel):
    num_perm: int = 128
    shingle_size: int = 5
    similarities: list[float] = Field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 0.95])


class LSHSettings(BaseModel):
    num_bands: int = 32
    rows_per_band: int = 4
    target_threshold: float = 0.5


class FeatureHashingSettings(BaseModel):
    n_buckets: int = 256
    bucket_counts: list[int] = Field(
        default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024, 4096]
    )
    alternate_sign: bool = True


class CorpusSettings(BaseModel):
    n_docs: int = 400
    n_near_duplicates: int = 40
    doc_words: int = 60
    vocabulary_size: int = 400
    edit_fraction: float = 0.04


class CategoricalSettings(BaseModel):
    n_rows: int = 20_000
    n_categories: int = 5_000
    n_informative: int = 200
    noise: float = 0.3


class FigureSettings(BaseModel):
    dpi: int = 120
    directory: str = "outputs/figures"

    @property
    def path(self) -> Path:
        """Absolute figure directory, resolved against the project root."""
        return PROJECT_ROOT / self.directory


class ResultSettings(BaseModel):
    directory: str = "outputs/results"

    @property
    def path(self) -> Path:
        """Absolute results directory, resolved against the project root."""
        return PROJECT_ROOT / self.directory


class Settings(BaseSettings):
    """The whole project's configuration, loaded from ``config/settings.yaml``."""

    model_config = SettingsConfigDict(
        yaml_file=CONFIG_PATH,
        env_prefix="HASHING101_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    seed: int = 20260903
    stream: StreamSettings = Field(default_factory=StreamSettings)
    hashtable: HashTableSettings = Field(default_factory=HashTableSettings)
    bloom: BloomSettings = Field(default_factory=BloomSettings)
    counting_bloom: CountingBloomSettings = Field(default_factory=CountingBloomSettings)
    hyperloglog: HyperLogLogSettings = Field(default_factory=HyperLogLogSettings)
    count_min: CountMinSettings = Field(default_factory=CountMinSettings)
    minhash: MinHashSettings = Field(default_factory=MinHashSettings)
    lsh: LSHSettings = Field(default_factory=LSHSettings)
    feature_hashing: FeatureHashingSettings = Field(default_factory=FeatureHashingSettings)
    corpus: CorpusSettings = Field(default_factory=CorpusSettings)
    categorical: CategoricalSettings = Field(default_factory=CategoricalSettings)
    figures: FigureSettings = Field(default_factory=FigureSettings)
    results: ResultSettings = Field(default_factory=ResultSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """YAML is the base; environment variables (``HASHING101_SEED=...``) win over it."""
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load (and cache) the project settings."""
    return Settings()
