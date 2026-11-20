import numpy as np

from hd_search_relevance.evaluation import ndcg_at_k, rmse


def test_rmse_known():
    assert abs(rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])) < 1e-12
    assert rmse([0.0, 0.0], [1.0, 1.0]) == 1.0


def test_ndcg_perfect_ranking():
    y_true = np.array([3.0, 2.0, 1.0])
    y_pred = np.array([10.0, 5.0, 1.0])
    groups = np.array(["q", "q", "q"])
    score = ndcg_at_k(y_true, y_pred, groups, k=3)
    assert abs(score - 1.0) < 1e-9


def test_ndcg_worst_ranking():
    y_true = np.array([3.0, 2.0, 1.0])
    y_pred = np.array([1.0, 5.0, 10.0])  # predicts reverse order
    groups = np.array(["q", "q", "q"])
    score = ndcg_at_k(y_true, y_pred, groups, k=3)
    assert score < 1.0
