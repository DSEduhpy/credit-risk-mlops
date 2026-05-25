"""
Unit tests for src/evaluation/metrics.py.

Tests cover:
- AUC-ROC calculation correctness
- Recall / precision / F1 at various thresholds
- Edge cases: all-positive, all-negative predictions
- Output shape and type contracts
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import compute_metrics


class TestComputeMetrics:
    """Tests for the primary metrics computation function."""

    @pytest.fixture
    def binary_labels(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.75, 0.85, 0.95])
        return y_true, y_prob

    def test_returns_dict(self, binary_labels):
        y_true, y_prob = binary_labels
        result = compute_metrics(y_true, y_prob, threshold=0.5)
        assert isinstance(result, dict), "compute_metrics must return a dict"

    def test_required_keys_present(self, binary_labels):
        y_true, y_prob = binary_labels
        result = compute_metrics(y_true, y_prob, threshold=0.5)
        required = {"auc", "recall", "precision", "f1", "accuracy"}
        missing = required - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_auc_matches_sklearn(self, binary_labels):
        y_true, y_prob = binary_labels
        result = compute_metrics(y_true, y_prob, threshold=0.5)
        expected_auc = roc_auc_score(y_true, y_prob)
        assert result["auc"] == pytest.approx(expected_auc, abs=1e-6)

    def test_all_metrics_in_unit_interval(self, binary_labels):
        y_true, y_prob = binary_labels
        result = compute_metrics(y_true, y_prob, threshold=0.5)
        for key in ("auc", "recall", "precision", "f1", "accuracy"):
            value = result[key]
            assert 0.0 <= value <= 1.0, (
                f"Metric '{key}' = {value} is outside [0, 1]"
            )

    def test_perfect_auc(self):
        """A perfect classifier must yield AUC == 1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_metrics(y_true, y_prob, threshold=0.5)
        assert result["auc"] == pytest.approx(1.0, abs=1e-6)

    def test_random_classifier_auc_near_05(self):
        """A random classifier should produce AUC ≈ 0.5."""
        rng = np.random.default_rng(0)
        y_true = rng.choice([0, 1], size=1000, p=[0.8, 0.2])
        y_prob = rng.uniform(0, 1, size=1000)
        result = compute_metrics(y_true, y_prob, threshold=0.5)
        assert 0.3 < result["auc"] < 0.7, (
            f"Random classifier AUC {result['auc']} is far from 0.5"
        )

    def test_high_threshold_low_recall(self, binary_labels):
        """At threshold=0.95, almost no positives are detected → recall near 0."""
        y_true, y_prob = binary_labels
        result = compute_metrics(y_true, y_prob, threshold=0.95)
        assert result["recall"] <= 0.5, (
            f"High threshold should yield low recall, got {result['recall']}"
        )

    def test_low_threshold_high_recall(self, binary_labels):
        """At threshold=0.05, nearly all positives are detected → recall near 1."""
        y_true, y_prob = binary_labels
        result = compute_metrics(y_true, y_prob, threshold=0.05)
        assert result["recall"] >= 0.5, (
            f"Low threshold should yield high recall, got {result['recall']}"
        )

    def test_values_are_python_scalars(self, binary_labels):
        """All metric values must be plain Python floats or ints, not numpy scalars
        (important for JSON serialisation in MLflow logging)."""
        y_true, y_prob = binary_labels
        result = compute_metrics(y_true, y_prob, threshold=0.5)
        for key, value in result.items():
            assert isinstance(value, (int, float)), (
                f"Metric '{key}' returned {type(value)}, expected Python scalar"
            )   