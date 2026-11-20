"""Command-line interface for training and prediction."""

from __future__ import annotations

import argparse
import json
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import save_artifact
from .config import AppConfig
from .data import build_test_frame, build_training_frame, load_raw_data
from .embeddings import load_embedding_model
from .evaluation import evaluate_predictions
from .features import add_clean_columns, add_lexical_features, build_feature_matrix
from .inference import load_and_predict
from .training import (
    cross_validate,
    fit_features_for_split,
    group_train_validation_split,
    mean_baseline_prediction,
    prepare_test_features,
    train_lightgbm,
    train_random_forest,
)


def train_command(args: argparse.Namespace) -> None:
    config = AppConfig()
    data_dir = Path(args.data_dir)
    artifact_path = Path(args.artifact_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_cached_features:
        print("Loading cached feature frames...")
        train_frame = pd.read_parquet("outputs/train_features.parquet")
        test_frame = pd.read_parquet("outputs/test_features.parquet")
        if train_frame["bert_semantic_title_sim"].isna().any():
            raise ValueError("Cached training features still contain missing BERT values")
        if test_frame["bert_semantic_title_sim"].isna().any():
            raise ValueError("Cached test features still contain missing BERT values")
    else:
        print("Loading data...")
        raw = load_raw_data(data_dir)
        train_frame = build_training_frame(raw)
        test_frame = build_test_frame(raw)

        print("Preparing text and lexical features...")
        train_frame = add_clean_columns(train_frame)
        train_frame = add_lexical_features(train_frame)
        test_frame = add_clean_columns(test_frame)
        test_frame = add_lexical_features(test_frame)

    print("Loading embedding model...")
    embedding_model = load_embedding_model(config.model.embedding_model_name)

    if not args.use_cached_features:
        print("Computing semantic features for training data...")
        from .features import add_bert_features

        train_frame = add_bert_features(train_frame, embedding_model, config.model.embedding_batch_size)
        print("Computing semantic features for test data...")
        test_frame = add_bert_features(test_frame, embedding_model, config.model.embedding_batch_size)

    print("Creating group-aware train/validation split...")
    train_split, val_split = group_train_validation_split(
        train_frame,
        config.model.group_column,
        config.model.test_size,
        config.model.random_state,
    )

    print("Fitting features for split...")
    X_train, y_train, X_val, y_val, vectorizer = fit_features_for_split(
        train_split, val_split, embedding_model, config.model
    )

    results: dict[str, Any] = {}

    # Baseline
    baseline_pred = mean_baseline_prediction(y_train, len(y_val))
    baseline_metrics = evaluate_predictions(
        y_val, baseline_pred, val_split[config.model.group_column].to_numpy()
    )
    results["baseline"] = baseline_metrics
    print("Baseline:", baseline_metrics)

    # Random Forest
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train, config.model.rf_params)
    rf_pred = rf_model.predict(X_val)
    rf_metrics = evaluate_predictions(y_val, rf_pred, val_split[config.model.group_column].to_numpy())
    results["random_forest"] = rf_metrics
    print("Random Forest:", rf_metrics)

    # LightGBM
    print("Training LightGBM...")
    lgbm_model = train_lightgbm(X_train, y_train, config.model.lgbm_params)
    lgbm_pred = lgbm_model.predict(X_val)
    lgbm_metrics = evaluate_predictions(y_val, lgbm_pred, val_split[config.model.group_column].to_numpy())
    results["lightgbm"] = lgbm_metrics
    print("LightGBM:", lgbm_metrics)

    # Cross-validation
    if not args.skip_cv:
        print("Running group-aware cross-validation for Random Forest...")
        rf_cv = cross_validate(train_frame, "random_forest", embedding_model, config.model, n_splits=5)
        results["random_forest_cv"] = rf_cv
        print("RF CV:", {k: v for k, v in rf_cv.items() if k.endswith("_mean")})

        print("Running group-aware cross-validation for LightGBM...")
        lgbm_cv = cross_validate(train_frame, "lightgbm", embedding_model, config.model, n_splits=5)
        results["lightgbm_cv"] = lgbm_cv
        print("LightGBM CV:", {k: v for k, v in lgbm_cv.items() if k.endswith("_mean")})

    # Final model on all training data
    print("Training final LightGBM model on all training data...")
    full_vectorizer_corpus = (
        train_frame["clean_product_title"].fillna("")
        + " "
        + train_frame["clean_product_description"].fillna("")
    )
    from .features import fit_tfidf

    final_vectorizer = fit_tfidf(
        full_vectorizer_corpus, config.model.tfidf_max_features, config.model.tfidf_stop_words
    )
    # Recompute TF-IDF features on full data with the final vectorizer.
    for column in ("tfidf_cos_title", "tfidf_cos_desc"):
        if column in train_frame.columns:
            del train_frame[column]
    train_frame = add_tfidf_features_with_vectorizer(train_frame, final_vectorizer)
    X_full = build_feature_matrix(train_frame, config.model.feature_names)
    y_full = train_frame[config.model.target_column].astype(float)
    final_model = train_lightgbm(X_full, y_full, config.model.lgbm_params)

    # Save artifact
    save_artifact(
        artifact_path,
        final_model,
        final_vectorizer,
        list(config.model.feature_names),
        config.model,
        metadata={
            "created_at": pd.Timestamp.utcnow().isoformat(),
            "embedding_model_name": config.model.embedding_model_name,
            "training_rows": len(train_frame),
            "validation_metrics": results,
        },
    )
    print(f"Saved model artifact to {artifact_path}")

    # Batch prediction on test
    print("Predicting test set...")
    X_test = prepare_test_features(test_frame, final_vectorizer, embedding_model, config.model)
    X_test = X_test.loc[:, config.model.feature_names]
    test_predictions = np.asarray(final_model.predict(X_test), dtype=float)
    test_predictions = np.clip(test_predictions, config.model.clip_min, config.model.clip_max)
    submission = pd.DataFrame(
        {
            config.model.id_column: test_frame[config.model.id_column].to_numpy(),
            config.model.target_column: test_predictions,
        }
    )
    submission_path = output_dir / config.paths.submission_filename
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"Saved metrics to {metrics_path}")


def add_tfidf_features_with_vectorizer(frame, vectorizer):
    from .features import add_tfidf_features

    return add_tfidf_features(frame, vectorizer)


def predict_command(args: argparse.Namespace) -> None:
    load_and_predict(args.input, args.artifact, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Home Depot search relevance CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the model")
    train_parser.add_argument("--data-dir", default="data")
    train_parser.add_argument("--artifact-path", default="artifacts/search_relevance_model.joblib")
    train_parser.add_argument("--output-dir", default="outputs")
    train_parser.add_argument("--skip-cv", action="store_true")
    train_parser.add_argument(
        "--use-cached-features",
        action="store_true",
        help="Load outputs/train_features.parquet and outputs/test_features.parquet.",
    )
    train_parser.set_defaults(func=train_command)

    predict_parser = subparsers.add_parser("predict", help="Run batch prediction")
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--artifact", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.set_defaults(func=predict_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
