"""Model artifact serialization helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn
import sys

from .config import ModelConfig


REQUIRED_KEYS = {"model", "tfidf_vectorizer", "feature_names", "config", "metadata"}


def save_artifact(
    path: Path | str,
    model: Any,
    vectorizer: Any,
    feature_names: list[str],
    config: ModelConfig,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a validated model artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "predict"):
        raise TypeError("model must implement predict")
    if not hasattr(vectorizer, "transform"):
        raise TypeError("tfidf_vectorizer must implement transform")

    artifact = {
        "model": model,
        "tfidf_vectorizer": vectorizer,
        "feature_names": list(feature_names),
        "config": {
            "random_state": config.random_state,
            "feature_names": list(config.feature_names),
            "clip_min": config.clip_min,
            "clip_max": config.clip_max,
            "embedding_model_name": config.embedding_model_name,
            "tfidf_max_features": config.tfidf_max_features,
        },
        "metadata": metadata
        or {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
            "feature_count": len(feature_names),
        },
    }
    joblib.dump(artifact, path)
    return path


def load_artifact(path: Path | str) -> dict[str, Any]:
    """Load and validate a model artifact."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError("Invalid artifact format: expected a dictionary")
    missing = REQUIRED_KEYS - set(artifact)
    if missing:
        raise ValueError(f"Artifact is missing keys: {sorted(missing)}")
    if not hasattr(artifact["model"], "predict"):
        raise ValueError("Artifact model does not implement predict")
    if not hasattr(artifact["tfidf_vectorizer"], "transform"):
        raise ValueError("Artifact TF-IDF vectorizer is not fitted")
    if artifact["feature_names"] != artifact["config"].get("feature_names"):
        raise ValueError("Artifact feature_names do not match config")
    return artifact
