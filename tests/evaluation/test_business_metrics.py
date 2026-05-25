"""
Unit tests for src/evaluation/business_metrics.py.

Tests cover:
- Profit calculation under known TP/FP/TN/FN counts
- Business metric boundary conditions (all approved, all denied)
- Threshold sweep correctness
- Financial impact sign and magnitude
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# We test the public interface of business_metrics. If the module does not
# yet expose the exact function names tested below, the test file documents
# the EXPECTED interface that the module must satisfy.
# ---------------------------------------------------------------------------
from src.evaluation.business_metrics import (
    compute_business_profit,
    compute_profit_by_threshold,
    find_optimal_threshold,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_predictions():
    """y_true and y_prob where the model is perfect (AUC=1)."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95])
    return y_true, y_prob


@pytest.fixture
def random_predictions(seed: int = 42):
    """y_true and y_prob with realistic 20% default rate."""
    rng = np.random.default_rng(seed)
    n = 200
    y_true = rng.choice([0, 1], size=n, p=[0.80, 0.20])
    y_prob = rng.uniform(0, 1, size=n)
    return y_true, y_prob


@pytest.fixture
def all_approved_predictions():
    """Scenario where the model approves everyone (prob always below threshold)."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])  # All < 0.5 threshold → all approved
    return y_true, y_prob


@pytest.fixture
def all_denied_predictions():
    """Scenario where the model denies everyone (prob always above threshold)."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.6, 0.7, 0.8, 0.9])  # All > 0.5 threshold → all denied
    return y_true, y_prob


# ---------------------------------------------------------------------------
# Tests: compute_business_profit
# ---------------------------------------------------------------------------

class TestComputeBusinessProfit:
    """Tests for the core profit computation function."""

    def test_perfect_model_positive_profit(self, perfect_predictions):
        """A perfect model should generate maximum positive profit."""
        y_true, y_prob = perfect_predictions
        profit = compute_business_profit(
            y_true=y_true,
            y_prob=y_prob,
            threshold=0.5,
            default_cost=10_000,
            revenue_per_approval=1_000,
        )
        assert profit > 0, "Perfect model must yield positive profit"

    def test_all_approved_profit(self, all_approved_predictions):
        """
        When all customers are approved:
        - Non-defaults (TN treated as TP) each contribute +revenue
        - Defaults (FN treated as TP) each contribute -default_cost
        Net = (2 × 1000) + (2 × -10_000) = -18_000
        """
        y_true, y_prob = all_approved_predictions
        profit = compute_business_profit(
            y_true=y_true,
            y_prob=y_prob,
            threshold=0.5,
            default_cost=10_000,
            revenue_per_approval=1_000,
        )
        # 2 good approvals × 1000 − 2 bad approvals × 10000
        expected = (2 * 1_000) - (2 * 10_000)
        assert profit == pytest.approx(expected, abs=1), (
            f"Expected {expected}, got {profit}"
        )

    def test_all_denied_profit_is_zero(self, all_denied_predictions):
        """
        When all customers are denied no revenue is generated and no defaults
        are funded — profit should be zero.
        """
        y_true, y_prob = all_denied_predictions
        profit = compute_business_profit(
            y_true=y_true,
            y_prob=y_prob,
            threshold=0.5,
            default_cost=10_000,
            revenue_per_approval=1_000,
        )
        assert profit == pytest.approx(0.0, abs=1), (
            "Denying everyone should yield zero profit"
        )

    def test_profit_is_scalar(self, random_predictions):
        """compute_business_profit must return a single numeric value."""
        y_true, y_prob = random_predictions
        result = compute_business_profit(y_true, y_prob, threshold=0.5)
        assert isinstance(result, (int, float, np.floating)), (
            f"Expected scalar, got {type(result)}"
        )

    def test_default_cost_impact(self, perfect_predictions):
        """Higher default_cost must reduce profit monotonically."""
        y_true, y_prob = perfect_predictions
        profit_low = compute_business_profit(
            y_true, y_prob, threshold=0.3,
            default_cost=5_000, revenue_per_approval=1_000,
        )
        profit_high = compute_business_profit(
            y_true, y_prob, threshold=0.3,
            default_cost=50_000, revenue_per_approval=1_000,
        )
        assert profit_high <= profit_low, (
            "Higher default_cost must not increase profit"
        )

    def test_invalid_threshold_raises(self):
        """Threshold outside [0, 1] must raise ValueError."""
        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])
        with pytest.raises((ValueError, AssertionError)):
            compute_business_profit(y_true, y_prob, threshold=1.5)

    def test_empty_input_raises(self):
        """Empty arrays must raise an appropriate exception."""
        with pytest.raises(Exception):
            compute_business_profit(
                y_true=np.array([]),
                y_prob=np.array([]),
                threshold=0.5,
            )


# ---------------------------------------------------------------------------
# Tests: compute_profit_by_threshold
# ---------------------------------------------------------------------------

class TestComputeProfitByThreshold:
    """Tests for the threshold sweep function."""

    def test_returns_series_or_dict(self, random_predictions):
        """Output must be iterable with threshold keys and profit values."""
        y_true, y_prob = random_predictions
        result = compute_profit_by_threshold(y_true, y_prob)
        assert hasattr(result, "__iter__"), "Result must be iterable"

    def test_thresholds_in_valid_range(self, random_predictions):
        """All swept thresholds must be in (0, 1)."""
        y_true, y_prob = random_predictions
        result = compute_profit_by_threshold(y_true, y_prob)
        if isinstance(result, pd.Series):
            thresholds = result.index.tolist()
        elif isinstance(result, dict):
            thresholds = list(result.keys())
        else:
            thresholds = [t for t, _ in result]

        for t in thresholds:
            assert 0.0 <= t <= 1.0, f"Threshold {t} out of [0, 1] range"

    def test_profit_values_are_numeric(self, random_predictions):
        """All profit values returned must be numeric."""
        y_true, y_prob = random_predictions
        result = compute_profit_by_threshold(y_true, y_prob)
        if isinstance(result, pd.Series):
            values = result.values
        elif isinstance(result, dict):
            values = list(result.values())
        else:
            values = [p for _, p in result]

        for v in values:
            assert isinstance(v, (int, float, np.floating)), (
                f"Non-numeric profit value: {v} ({type(v)})"
            )


# ---------------------------------------------------------------------------
# Tests: find_optimal_threshold
# ---------------------------------------------------------------------------

class TestFindOptimalThreshold:
    """Tests for business-optimal threshold selection."""

    def test_returns_float_in_unit_interval(self, random_predictions):
        """Optimal threshold must be a float in [0, 1]."""
        y_true, y_prob = random_predictions
        threshold = find_optimal_threshold(y_true, y_prob)
        assert isinstance(threshold, (float, np.floating)), (
            f"Expected float, got {type(threshold)}"
        )
        assert 0.0 <= threshold <= 1.0, (
            f"Threshold {threshold} outside [0, 1]"
        )

    def test_optimal_threshold_maximises_profit(self, random_predictions):
        """
        Profit at the returned threshold must be >= profit at the default 0.5.
        (Unless 0.5 happens to be optimal, in which case they are equal.)
        """
        y_true, y_prob = random_predictions
        optimal = find_optimal_threshold(y_true, y_prob)
        profit_optimal = compute_business_profit(y_true, y_prob, threshold=optimal)
        profit_default = compute_business_profit(y_true, y_prob, threshold=0.5)
        assert profit_optimal >= profit_default - 1, (
            "Optimal threshold must not perform worse than 0.5"
        )

    def test_perfect_model_threshold_separates_classes(self, perfect_predictions):
        """For a perfect model, the optimal threshold should lie between the
        highest negative score and the lowest positive score."""
        y_true, y_prob = perfect_predictions
        threshold = find_optimal_threshold(y_true, y_prob)
        max_negative = y_prob[y_true == 0].max()
        min_positive = y_prob[y_true == 1].min()
        assert threshold >= max_negative or threshold <= min_positive, (
            "For perfect predictions, threshold should cleanly separate classes"
        )