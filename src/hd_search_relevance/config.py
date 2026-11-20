"""Configuration for the Home Depot search relevance project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Filesystem locations for input data."""

    data_dir: Path = Path("data")
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    descriptions_file: str = "product_descriptions.csv"
    attributes_file: str = "attributes.csv"
    sample_submission_file: str = "sample_submission.csv"

    def path(self, filename: str) -> Path:
        return self.data_dir / filename


@dataclass(frozen=True)
class ModelConfig:
    """Model and feature configuration."""

    random_state: int = 42
    test_size: float = 0.2
    group_column: str = "search_term"
    target_column: str = "relevance"
    id_column: str = "id"
    product_uid_column: str = "product_uid"

    feature_names: tuple[str, ...] = (
        "search_len",
        "word_match_title",
        "word_match_desc",
        "word_match_brand",
        "ratio_title",
        "ratio_desc",
        "ratio_brand",
        "jaccard_title",
        "jaccard_desc",
        "jaccard_brand",
        "tfidf_cos_title",
        "tfidf_cos_desc",
        "bert_semantic_title_sim",
    )

    rf_params: dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 12,
        "random_state": 42,
        "n_jobs": -1,
    })
    lgbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 7,
        "num_leaves": 31,
        "min_child_samples": 20,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })

    tfidf_max_features: int = 15000
    tfidf_stop_words: str = "english"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    embedding_normalize: bool = True

    clip_min: float = 1.0
    clip_max: float = 3.0


@dataclass(frozen=True)
class PathConfig:
    """Output locations."""

    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("models")
    outputs_dir: Path = Path("outputs")
    model_filename: str = "search_relevance_model.joblib"
    submission_filename: str = "submission.csv"


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
