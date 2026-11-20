"""Feature engineering for search relevance."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import ModelConfig
from .embeddings import rowwise_cosine
from .preprocessing import clean_text

DEFAULT_CONFIG = ModelConfig()


def add_clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add canonical cleaned text columns."""
    result = frame.copy()
    result["clean_search_term"] = result["search_term"].map(clean_text)
    result["clean_product_title"] = result["product_title"].map(clean_text)
    result["clean_product_description"] = result["product_description"].map(clean_text)
    result["clean_product_brand"] = result["product_brand"].map(clean_text)
    return result


def _token_set(text: str) -> set[str]:
    return set(str(text).split())


def add_lexical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add token-overlap features used by the baseline and advanced models."""
    result = frame.copy()
    result["search_len"] = result["clean_search_term"].map(lambda x: len(str(x).split()))

    title_matches = []
    desc_matches = []
    brand_matches = []
    for _, row in result.iterrows():
        search_tokens = _token_set(row["clean_search_term"])
        title_tokens = _token_set(row["clean_product_title"])
        desc_tokens = _token_set(row["clean_product_description"])
        brand_tokens = _token_set(row["clean_product_brand"])
        title_matches.append(len(search_tokens & title_tokens))
        desc_matches.append(len(search_tokens & desc_tokens))
        brand_matches.append(len(search_tokens & brand_tokens))

    result["word_match_title"] = title_matches
    result["word_match_desc"] = desc_matches
    result["word_match_brand"] = brand_matches

    eps = 1e-5
    result["ratio_title"] = result["word_match_title"] / (result["search_len"] + eps)
    result["ratio_desc"] = result["word_match_desc"] / (result["search_len"] + eps)
    result["ratio_brand"] = result["word_match_brand"] / (result["search_len"] + eps)

    def jaccard(left: set[str], right: set[str]) -> float:
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    result["jaccard_title"] = [
        jaccard(_token_set(a), _token_set(b))
        for a, b in zip(result["clean_search_term"], result["clean_product_title"])
    ]
    result["jaccard_desc"] = [
        jaccard(_token_set(a), _token_set(b))
        for a, b in zip(result["clean_search_term"], result["clean_product_description"])
    ]
    result["jaccard_brand"] = [
        jaccard(_token_set(a), _token_set(b))
        for a, b in zip(result["clean_search_term"], result["clean_product_brand"])
    ]
    return result


def fit_tfidf(
    corpus: Iterable[str],
    max_features: int = DEFAULT_CONFIG.tfidf_max_features,
    stop_words: str | None = DEFAULT_CONFIG.tfidf_stop_words,
) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on a training corpus only."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    vectorizer.fit(list(corpus))
    return vectorizer


def tfidf_cosine_similarity(
    vectorizer: TfidfVectorizer,
    left_texts: pd.Series | list[str],
    right_texts: pd.Series | list[str],
) -> np.ndarray:
    """Compute row-wise cosine similarity using an already fitted vectorizer."""
    left_matrix = vectorizer.transform(list(left_texts))
    right_matrix = vectorizer.transform(list(right_texts))
    # TfidfVectorizer uses L2 normalization by default, so the dot product is
    # the cosine similarity.
    similarities = np.asarray((left_matrix.multiply(right_matrix)).sum(axis=1)).ravel()
    return similarities.astype(np.float64)


def add_tfidf_features(
    frame: pd.DataFrame,
    vectorizer: TfidfVectorizer,
) -> pd.DataFrame:
    """Add TF-IDF cosine similarity features."""
    result = frame.copy()
    result["tfidf_cos_title"] = tfidf_cosine_similarity(
        vectorizer, result["clean_search_term"], result["clean_product_title"]
    )
    result["tfidf_cos_desc"] = tfidf_cosine_similarity(
        vectorizer, result["clean_search_term"], result["clean_product_description"]
    )
    return result


def add_bert_features(
    frame: pd.DataFrame,
    embedding_model,
    batch_size: int = DEFAULT_CONFIG.embedding_batch_size,
) -> pd.DataFrame:
    """Add Sentence-BERT semantic similarity between search term and title."""
    from .embeddings import encode_texts

    result = frame.copy()
    search_embeddings = encode_texts(
        embedding_model, result["clean_search_term"].tolist(), batch_size=batch_size
    )
    title_embeddings = encode_texts(
        embedding_model, result["clean_product_title"].tolist(), batch_size=batch_size
    )
    result["bert_semantic_title_sim"] = rowwise_cosine(search_embeddings, title_embeddings)
    return result


def build_feature_matrix(frame: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    """Return a DataFrame containing exactly the configured feature columns."""
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return frame.loc[:, list(feature_names)].copy()
