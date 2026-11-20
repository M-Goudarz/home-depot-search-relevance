from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor

from hd_search_relevance.artifacts import load_artifact, save_artifact
from hd_search_relevance.config import ModelConfig


def test_artifact_roundtrip(tmp_path: Path):
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    vectorizer.fit(["alpha beta", "gamma delta"])
    config = ModelConfig(feature_names=("search_len",))
    artifact_path = tmp_path / "model.joblib"
    save_artifact(artifact_path, model, vectorizer, ["search_len"], config)
    loaded = load_artifact(artifact_path)
    assert loaded["feature_names"] == ["search_len"]
    assert hasattr(loaded["model"], "predict")
    assert hasattr(loaded["tfidf_vectorizer"], "transform")
