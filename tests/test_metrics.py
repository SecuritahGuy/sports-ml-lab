"""Tests for the evaluation metrics module."""

import numpy as np
import pytest

from sportslab.evaluation.metrics import compute_classification_metrics


class TestComputeClassificationMetrics:
    def test_perfect_prediction(self):
        y_true = np.array([1, 0, 1, 0])
        y_proba = np.array([1.0, 0.0, 1.0, 0.0])
        m = compute_classification_metrics(y_true, y_proba)
        assert m["log_loss"] == pytest.approx(0.0, abs=1e-6)
        assert m["brier_score"] == pytest.approx(0.0, abs=1e-6)
        assert m["accuracy"] == 1.0
        assert m["roc_auc"] == 1.0

    def test_random_prediction(self):
        n = 10000
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=n)
        y_proba = np.full(n, 0.5)
        m = compute_classification_metrics(y_true, y_proba)
        assert m["accuracy"] == pytest.approx(0.5, abs=0.02)
        assert m["roc_auc"] == pytest.approx(0.5, abs=0.02)

    def test_all_same_class(self):
        y_true = np.ones(10, dtype=int)
        y_proba = np.full(10, 0.8)
        m = compute_classification_metrics(y_true, y_proba)
        assert m["roc_auc"] is None
        assert "calibration_buckets" in m

    def test_calibration_buckets_present(self):
        np.random.seed(0)
        y_true = np.random.randint(0, 2, size=200)
        y_proba = np.random.rand(200)
        m = compute_classification_metrics(y_true, y_proba)
        buckets = m["calibration_buckets"]
        assert len(buckets) > 0
        for bucket_label, b in buckets.items():
            assert b["count"] > 0
            assert 0 <= b["mean_predicted_prob"] <= 1
            assert 0 <= b["mean_actual_rate"] <= 1

    def test_partial_calibration_coverage(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_proba = np.array([0.95, 0.92, 0.97, 0.93, 0.99])
        m = compute_classification_metrics(y_true, y_proba)
        buckets = m["calibration_buckets"]
        keys = list(buckets.keys())
        assert any("[0.9, 1.0)" in k for k in keys)
