import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

from hd_search_relevance.artifacts import save_artifact
from hd_search_relevance.config import ModelConfig
from hd_search_relevance.inference import predict_frame, score_candidates


class FakeEmbeddingModel:
    def encode(self, texts, batch_size=8, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        return np.ones((len(texts), 4), dtype=np.float32)


def test_predict_frame_shape_finite(tmp_path):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(["drill bit", "hammer drill"])
    model = DummyRegressor(strategy="mean")
    model.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
    config = ModelConfig(feature_names=("search_len",))
    artifact_path = tmp_path / "model.joblib"
    save_artifact(artifact_path, model, vectorizer, ["search_len"], config)

    from hd_search_relevance.artifacts import load_artifact

    artifact = load_artifact(artifact_path)
    frame = pd.DataFrame(
        {
            "product_title": ["drill bit set", "hammer drill"],
            "search_term": ["drill", "drill"],
            "product_description": ["includes drill", "heavy duty"],
            "product_brand": ["DeWalt", "Makita"],
        }
    )
    scored = predict_frame(frame, artifact, embedding_model=FakeEmbeddingModel(), clip=True)
    assert "predicted_relevance" in scored.columns
    assert scored["predicted_relevance"].shape == (2,)
    assert np.all(np.isfinite(scored["predicted_relevance"]))
    assert ((scored["predicted_relevance"] >= 1.0) & (scored["predicted_relevance"] <= 3.0)).all()


def test_score_candidates_sorts_and_skips_empty_titles(tmp_path):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(["drill bit", "hammer drill"])
    model = DummyRegressor(strategy="mean")
    model.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
    config = ModelConfig(feature_names=("search_len",))
    artifact_path = tmp_path / "model.joblib"
    save_artifact(artifact_path, model, vectorizer, ["search_len"], config)

    from hd_search_relevance.artifacts import load_artifact

    artifact = load_artifact(artifact_path)
    results = score_candidates(
        "drill",
        [
            {"title": "Drill bit set", "description": "bits", "brand": "DeWalt"},
            {"title": "", "description": "ignored", "brand": "Unknown"},
            {"title": "Hammer drill", "description": "corded", "brand": "Makita"},
        ],
        artifact,
        embedding_model=FakeEmbeddingModel(),
    )
    assert list(results["rank"]) == [1, 2]
    assert results["predicted_relevance"].is_monotonic_decreasing


def test_score_candidates_rejects_empty_query_and_titles():
    artifact = {
        "feature_names": ["search_len"],
        "config": {"clip_min": 1.0, "clip_max": 3.0},
        "tfidf_vectorizer": None,
        "model": None,
    }
    with np.testing.assert_raises(ValueError):
        score_candidates("", [], artifact)
    with np.testing.assert_raises(ValueError):
        score_candidates("drill", [{"title": ""}], artifact)
