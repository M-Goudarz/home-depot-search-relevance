"""Model training utilities."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

from .config import ModelConfig
from .evaluation import evaluate_predictions, group_kfold_splits
from .features import (
    add_bert_features,
    add_clean_columns,
    add_lexical_features,
    add_tfidf_features,
    build_feature_matrix,
    fit_tfidf,
)


def _ensure_base_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "clean_search_term" not in result.columns:
        from .features import add_clean_columns

        result = add_clean_columns(result)
    if "search_len" not in result.columns:
        result = add_lexical_features(result)
    return result


def group_train_validation_split(
    frame: pd.DataFrame,
    group_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows while keeping all rows for each query in one partition."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(
        splitter.split(frame, groups=frame[group_column].to_numpy())
    )
    return (
        frame.iloc[train_idx].reset_index(drop=True),
        frame.iloc[val_idx].reset_index(drop=True),
    )


def fit_features_for_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    embedding_model: Any,
    config: ModelConfig,
    vectorizer=None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, object]:
    """Fit TF-IDF on train and build feature matrices for both partitions."""
    train_df = _ensure_base_features(train_df)
    val_df = _ensure_base_features(val_df)

    if "bert_semantic_title_sim" not in train_df.columns:
        train_df = add_bert_features(train_df, embedding_model, config.embedding_batch_size)
    if "bert_semantic_title_sim" not in val_df.columns:
        val_df = add_bert_features(val_df, embedding_model, config.embedding_batch_size)

    if vectorizer is None:
        corpus = (
            train_df["clean_product_title"].fillna("")
            + " "
            + train_df["clean_product_description"].fillna("")
        )
        vectorizer = fit_tfidf(corpus, config.tfidf_max_features, config.tfidf_stop_words)

    # Recompute TF-IDF features from the (possibly fold-specific) vectorizer.
    for frame in (train_df, val_df):
        for column in ("tfidf_cos_title", "tfidf_cos_desc"):
            if column in frame.columns:
                del frame[column]
    train_df = add_tfidf_features(train_df, vectorizer)
    val_df = add_tfidf_features(val_df, vectorizer)

    X_train = build_feature_matrix(train_df, config.feature_names)
    y_train = train_df[config.target_column].astype(float)
    X_val = build_feature_matrix(val_df, config.feature_names)
    y_val = val_df[config.target_column].astype(float)
    return X_train, y_train, X_val, y_val, vectorizer


def train_random_forest(X: pd.DataFrame, y: pd.Series, params: dict[str, Any]) -> RandomForestRegressor:
    model = RandomForestRegressor(**params)
    model.fit(X, y)
    return model


def train_lightgbm(X: pd.DataFrame, y: pd.Series, params: dict[str, Any]) -> LGBMRegressor:
    model = LGBMRegressor(**params)
    model.fit(X, y)
    return model


def mean_baseline_prediction(train_y: pd.Series, size: int) -> np.ndarray:
    return np.full(size, float(np.mean(train_y)), dtype=float)


def cross_validate(
    full_df: pd.DataFrame,
    model_type: str,
    embedding_model: Any,
    config: ModelConfig,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Run leakage-safe, group-aware cross-validation."""
    full_df = _ensure_base_features(full_df)
    if "bert_semantic_title_sim" not in full_df.columns:
        full_df = add_bert_features(full_df, embedding_model, config.embedding_batch_size)

    groups = full_df[config.group_column].to_numpy()
    fold_results: list[dict[str, float]] = []
    for fold, (train_idx, val_idx) in enumerate(
        group_kfold_splits(groups, n_splits), start=1
    ):
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)

        # Drop fold-specific TF-IDF features so they are refit on the train fold.
        for column in ("tfidf_cos_title", "tfidf_cos_desc"):
            if column in train_df.columns:
                del train_df[column]
            if column in val_df.columns:
                del val_df[column]

        X_train, y_train, X_val, y_val, _ = fit_features_for_split(
            train_df, val_df, embedding_model, config
        )

        if model_type == "random_forest":
            model = train_random_forest(X_train, y_train, config.rf_params)
        elif model_type == "lightgbm":
            model = train_lightgbm(X_train, y_train, config.lgbm_params)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        predictions = model.predict(X_val)
        fold_groups = val_df[config.group_column].to_numpy()
        metrics = evaluate_predictions(y_val, predictions, fold_groups)
        metrics["fold"] = fold
        fold_results.append(metrics)

    metric_names = [name for name in fold_results[0] if name != "fold"]
    summary: dict[str, Any] = {"folds": fold_results}
    for name in metric_names:
        values = np.asarray([fold[name] for fold in fold_results], dtype=float)
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_std"] = float(np.std(values, ddof=0))
    return summary


def prepare_test_features(
    test_df: pd.DataFrame,
    vectorizer,
    embedding_model: Any,
    config: ModelConfig,
) -> pd.DataFrame:
    """Build the final feature matrix for test data."""
    frame = _ensure_base_features(test_df)
    if "bert_semantic_title_sim" not in frame.columns:
        frame = add_bert_features(frame, embedding_model, config.embedding_batch_size)
    # Remove stale TF-IDF columns before recomputing with the fitted vectorizer.
    for column in ("tfidf_cos_title", "tfidf_cos_desc"):
        if column in frame.columns:
            del frame[column]
    frame = add_tfidf_features(frame, vectorizer)
    return build_feature_matrix(frame, config.feature_names)
