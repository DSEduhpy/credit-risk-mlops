"""
Integration tests for the Credit Risk MLOps pipeline.

These tests exercise multiple modules together to verify that data flows
correctly through the pipeline stages. They do NOT run the full DVC pipeline
(that is validated in CI) but they do test the Python interfaces in sequence.

Marked as ``integration`` — excluded from the pre-commit hook quick-check
but included in the full CI run.

Tests cover:
- clean → feature_engineering → validate → split data flow
- Metrics computed on model output match expected contracts
- Business metrics and statistical metrics are consistent
- Logger emits structured records without crashing pipeline code
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


class TestCleaningToFeatureEngineering:
    """Integration: cleaning output feeds correctly into feature engineering."""

    def test_clean_then_engineer_produces_required_features(
        self, raw_dataframe: pd.DataFrame
    ):
        from src.processing.cleaning import clean_data
        from src.processing.feature_engineering import engineer_features
        from src.config import settings

        cleaned = clean_data(raw_dataframe.copy())
        features = engineer_features(cleaned)

        required = settings.validation.required_feature_columns
        missing = [c for c in required if c not in features.columns]
        assert not missing, (
            f"After clean→engineer, missing required columns: {missing}"
        )

    def test_pipeline_output_has_no_inf_values(self, raw_dataframe: pd.DataFrame):
        from src.processing.cleaning import clean_data
        from src.processing.feature_engineering import engineer_features

        cleaned = clean_data(raw_dataframe.copy())
        features = engineer_features(cleaned)

        numeric_cols = features.select_dtypes(include=[np.number]).columns
        has_inf = np.isinf(features[numeric_cols]).any().any()
        assert not has_inf, "Pipeline produced infinite values in feature matrix"

    def test_pipeline_output_target_is_binary(self, raw_dataframe: pd.DataFrame):
        from src.processing.cleaning import clean_data
        from src.processing.feature_engineering import engineer_features

        cleaned = clean_data(raw_dataframe.copy())
        features = engineer_features(cleaned)

        if "default" in features.columns:
            unique_vals = set(features["default"].dropna().unique())
            assert unique_vals <= {0, 1}, (
                f"Target column contains non-binary values: {unique_vals}"
            )


class TestFeatureEngineeringToSplit:
    """Integration: engineer_features output feeds correctly into split_data."""

    def test_split_after_engineering_preserves_all_rows(
        self, raw_dataframe: pd.DataFrame
    ):
        from src.processing.cleaning import clean_data
        from src.processing.feature_engineering import engineer_features
        from src.modeling.data import split_data

        cleaned = clean_data(raw_dataframe.copy())
        features = engineer_features(cleaned)
        X_train, X_test, y_train, y_test = split_data(features)

        total_after_split = len(X_train) + len(X_test)
        assert total_after_split == len(features), (
            f"Rows lost during split: {len(features)} → {total_after_split}"
        )

    def test_split_target_distribution_close_to_original(
        self, raw_dataframe: pd.DataFrame
    ):
        from src.processing.cleaning import clean_data
        from src.processing.feature_engineering import engineer_features
        from src.modeling.data import split_data

        cleaned = clean_data(raw_dataframe.copy())
        features = engineer_features(cleaned)
        X_train, X_test, y_train, y_test = split_data(features, stratify=True)

        if "default" in features.columns:
            original_rate = features["default"].mean()
            train_rate = y_train.mean()
            assert abs(train_rate - original_rate) <= 0.15, (
                f"Stratification failed: original={original_rate:.2f}, "
                f"train={train_rate:.2f}"
            )


class TestMetricsConsistency:
    """Integration: statistical and business metrics are internally consistent."""

    def test_high_auc_correlates_with_positive_profit(
        self, feature_dataframe: pd.DataFrame
    ):
        """
        A model that assigns high scores to defaults should produce
        both high AUC and positive profit (relative to random baseline).
        """
        from src.evaluation.metrics import compute_metrics
        from src.evaluation.business_metrics import compute_business_profit

        y_true = feature_dataframe["default"].values
        # Near-perfect predictions
        rng = np.random.default_rng(42)
        y_prob_good = np.where(
            y_true == 1,
            rng.uniform(0.65, 0.95, size=len(y_true)),
            rng.uniform(0.05, 0.35, size=len(y_true)),
        )
        # Random predictions
        y_prob_random = rng.uniform(0, 1, size=len(y_true))

        metrics_good = compute_metrics(y_true, y_prob_good, threshold=0.5)
        metrics_random = compute_metrics(y_true, y_prob_random, threshold=0.5)

        profit_good = compute_business_profit(y_true, y_prob_good, threshold=0.5)
        profit_random = compute_business_profit(y_true, y_prob_random, threshold=0.5)

        assert metrics_good["auc"] > metrics_random["auc"], (
            "Near-perfect model must have higher AUC than random"
        )
        assert profit_good >= profit_random - 100, (
            "Near-perfect model must not have substantially lower profit than random"
        )


class TestLoggerIntegration:
    """Integration: logger produces structured output without breaking pipeline."""

    def test_pipeline_logger_does_not_raise(self):
        from src.logger import pipeline_logger, StageTimer

        log = pipeline_logger("test_stage")
        log.info("Integration test message")
        log.warning("Integration test warning")

        with StageTimer("test_stage"):
            _ = 1 + 1  # No-op operation

    def test_inference_logger_does_not_raise(self):
        from src.logger import log_inference

        log_inference(
            request_id="test-req-001",
            model="xgboost",
            prediction=0.42,
            latency_ms=8.7,
            threshold=0.5,
        )

    def test_drift_logger_does_not_raise(self):
        from src.logger import log_drift_alert

        log_drift_alert(
            feature="annual_inc",
            metric="PSI",
            value=0.25,
            threshold=0.20,
            severity="CRITICAL",
        )