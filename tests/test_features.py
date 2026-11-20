import numpy as np
import pandas as pd

from hd_search_relevance.features import (
    add_lexical_features,
    build_feature_matrix,
    fit_tfidf,
    tfidf_cosine_similarity,
)
from hd_search_relevance.config import ModelConfig


def test_lexical_features_counts():
    frame = pd.DataFrame(
        {
            "search_term": ["drill bit"],
            "product_title": ["drill bit set"],
            "product_description": ["includes drill and bit"],
            "product_brand": ["DeWalt"],
        }
    )
    from hd_search_relevance.features import add_clean_columns

    frame = add_clean_columns(frame)
    frame = add_lexical_features(frame)
    assert frame.loc[0, "search_len"] == 2
    assert frame.loc[0, "word_match_title"] == 2
    assert frame.loc[0, "word_match_desc"] == 2
    assert 0.0 <= frame.loc[0, "jaccard_title"] <= 1.0


def test_tfidf_fit_transform_separation():
    train_texts = ["drill bit", "hammer drill"]
    test_texts = ["drill", "unknown token"]
    vectorizer = fit_tfidf(train_texts, max_features=10, stop_words=None)
    # Vocabulary should come from train only.
    assert "unknown" not in vectorizer.vocabulary_
    similarities = tfidf_cosine_similarity(vectorizer, test_texts, test_texts)
    assert similarities.shape == (2,)
    assert np.all(np.isfinite(similarities))


def test_feature_matrix_order():
    config = ModelConfig()
    frame = pd.DataFrame({name: [1.0] for name in config.feature_names})
    matrix = build_feature_matrix(frame, config.feature_names)
    assert list(matrix.columns) == list(config.feature_names)
    assert matrix.shape == (1, len(config.feature_names))
