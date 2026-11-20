"""Inference helpers for saved search relevance models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import load_artifact
from .config import ModelConfig
from .embeddings import load_embedding_model
from .features import (
    add_bert_features,
    add_clean_columns,
    add_lexical_features,
    add_tfidf_features,
    build_feature_matrix,
)
from .training import prepare_test_features


INPUT_COLUMNS = ["product_title", "search_term"]


def _validate_input(frame: pd.DataFrame) -> None:
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Inference input is missing columns: {missing}")
    if len(frame) == 0:
        raise ValueError("Inference input is empty")


def predict_frame(
    frame: pd.DataFrame,
    artifact: dict[str, Any],
    embedding_model: Any | None = None,
    clip: bool = True,
) -> pd.DataFrame:
    """Predict relevance scores for a DataFrame of query-product pairs."""
    _validate_input(frame)
    result = frame.copy().reset_index(drop=True)
    for column, default in (
        ("product_description", ""),
        ("product_brand", "Unknown"),
        ("product_uid", -1),
    ):
        if column not in result.columns:
            result[column] = default

    if embedding_model is None:
        model_name = artifact["config"].get("embedding_model_name", ModelConfig().embedding_model_name)
        embedding_model = load_embedding_model(model_name)

    config = ModelConfig()
    config = ModelConfig(
        clip_min=artifact["config"].get("clip_min", config.clip_min),
        clip_max=artifact["config"].get("clip_max", config.clip_max),
        embedding_model_name=artifact["config"].get("embedding_model_name", config.embedding_model_name),
        feature_names=tuple(artifact["feature_names"]),
    )

    features = prepare_test_features(result, artifact["tfidf_vectorizer"], embedding_model, config)
    # Ensure exact saved feature order.
    features = features.loc[:, artifact["feature_names"]]
    predictions = np.asarray(artifact["model"].predict(features), dtype=float)

    if clip:
        predictions = np.clip(predictions, config.clip_min, config.clip_max)

    if not np.all(np.isfinite(predictions)):
        raise ValueError("Model produced non-finite predictions")

    result["predicted_relevance"] = predictions
    return result


def rank_candidates(
    frame: pd.DataFrame,
    artifact: dict[str, Any],
    embedding_model: Any | None = None,
    query_column: str = "search_term",
) -> pd.DataFrame:
    """Predict and rank candidate products within each query."""
    scored = predict_frame(frame, artifact, embedding_model=embedding_model, clip=True)
    scored["_score"] = scored["predicted_relevance"]
    scored["rank"] = (
        scored.groupby(query_column)["_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return scored.drop(columns=["_score"])


def load_and_predict(
    input_path: Path | str,
    artifact_path: Path | str,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """CLI-friendly prediction from CSV to CSV."""
    frame = pd.read_csv(input_path)
    artifact = load_artifact(artifact_path)
    scored = predict_frame(frame, artifact)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_path, index=False)
    return scored
