"""Evaluation metrics for regression and ranking."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(mean_absolute_error(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)))


def _dcg(relevances: np.ndarray) -> float:
    if relevances.size == 0:
        return 0.0
    positions = np.arange(1, relevances.size + 1)
    discounts = np.log2(positions + 1)
    gains = np.power(2.0, relevances) - 1.0
    return float(np.sum(gains / discounts))


def ndcg_at_k(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    groups: Iterable[object],
    k: int = 5,
) -> float:
    """Compute mean NDCG@k over query groups."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)
    if not (y_true.shape == y_pred.shape == groups.shape):
        raise ValueError("y_true, y_pred, and groups must have the same shape")

    scores: list[float] = []
    for group in np.unique(groups):
        mask = groups == group
        truth = y_true[mask]
        pred = y_pred[mask]
        if truth.size == 0:
            continue
        order = np.argsort(pred)[::-1][:k]
        dcg = _dcg(truth[order][:k])
        ideal_order = np.argsort(truth)[::-1][:k]
        idcg = _dcg(truth[ideal_order][:k])
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def evaluate_predictions(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    groups: Iterable[object],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Return regression and ranking metrics for aligned predictions."""
    result = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }
    for k in ks:
        result[f"ndcg_at_{k}"] = ndcg_at_k(y_true, y_pred, groups, k=k)
    return result


def group_kfold_splits(
    groups: Iterable[object], n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return train/validation index splits that keep each group intact."""
    groups = np.asarray(groups)
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups))
    return [(train_idx, val_idx) for train_idx, val_idx in splitter.split(dummy, groups=groups)]
