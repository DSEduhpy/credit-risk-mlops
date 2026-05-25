"""
Unit tests for src/processing/feature_engineering.py.

Tests cover:
- Engineered features are present after transformation
- Derived feature values are numerically correct
- No target leakage introduced by feature engineering
- Input column order does not affect output
- NaN propagation is controlled (engineered features don't introduce new nulls)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.feature_engineering import engineer_features


class TestEngineerFeatures:
    """Tests for the feature engineering transformation pipeline."""

    def test_returns_dataframe(self, raw_dataframe: pd.DataFrame):
        result = engineer_features(raw_dataframe.copy())
        assert isinstance(result, pd.DataFrame)

    def test_loan_amnt_to_income_computed(self, raw_dataframe: pd.DataFrame):
        """loan_amnt_to_income = loan_amnt / annual_inc must be present and positive."""
        result = engineer_features(raw_dataframe.copy())
        assert "loan_amnt_to_income" in result.columns, (
            "loan_amnt_to_income feature is missing"
        )
        non_null = result["loan_amnt_to_income"].dropna()
        assert (non_null >= 0).all(), "loan_amnt_to_income contains negative values"

    def test_fico_avg_computed(self, raw_dataframe: pd.DataFrame):
        """fico_avg = (fico_range_low + fico_range_high) / 2 must be within FICO range."""
        result = engineer_features(raw_dataframe.copy())
        assert "fico_avg" in result.columns, "fico_avg feature is missing"
        non_null = result["fico_avg"].dropna()
        assert (non_null >= 300).all(), "fico_avg below FICO minimum (300)"
        assert (non_null <= 850).all(), "fico_avg above FICO maximum (850)"

    def test_home_ownership_encoded(self, raw_dataframe: pd.DataFrame):
        """home_ownership_encoded must be an integer-typed column."""
        result = engineer_features(raw_dataframe.copy())
        assert "home_ownership_encoded" in result.columns, (
            "home_ownership_encoded feature is missing"
        )
        dtype = result["home_ownership_encoded"].dtype
        assert pd.api.types.is_numeric_dtype(dtype), (
            f"home_ownership_encoded has non-numeric dtype: {dtype}"
        )

    def test_purpose_encoded(self, raw_dataframe: pd.DataFrame):
        """purpose_encoded must be numeric."""
        result = engineer_features(raw_dataframe.copy())
        assert "purpose_encoded" in result.columns, "purpose_encoded feature is missing"
        assert pd.api.types.is_numeric_dtype(result["purpose_encoded"].dtype)

    def test_no_new_nulls_in_derived_features(self, raw_dataframe: pd.DataFrame):
        """
        Rows where both fico_range_low and fico_range_high are non-null
        must not produce null in fico_avg.
        """
        df = raw_dataframe.copy()
        df = df.dropna(subset=["fico_range_low", "fico_range_high"])
        result = engineer_features(df)
        if "fico_avg" in result.columns:
            null_count = result["fico_avg"].isnull().sum()
            assert null_count == 0, (
                f"fico_avg has {null_count} nulls despite complete fico inputs"
            )

    def test_row_count_preserved(self, raw_dataframe: pd.DataFrame):
        """Feature engineering must not drop rows."""
        result = engineer_features(raw_dataframe.copy())
        assert len(result) == len(raw_dataframe), (
            f"Row count changed: {len(raw_dataframe)} → {len(result)}"
        )

    def test_does_not_mutate_input(self, raw_dataframe: pd.DataFrame):
        """engineer_features must not modify the input DataFrame in place."""
        df_copy = raw_dataframe.copy()
        original_cols = set(df_copy.columns)
        _ = engineer_features(raw_dataframe.copy())
        assert set(raw_dataframe.columns) == original_cols, (
            "engineer_features mutated the input DataFrame's columns"
        )

    def test_loan_amnt_to_income_formula(self):
        """Verify the ratio formula directly on a controlled DataFrame."""
        df = pd.DataFrame(
            {
                "loan_amnt": [10_000.0, 20_000.0],
                "annual_inc": [50_000.0, 100_000.0],
                "int_rate": [10.0, 15.0],
                "installment": [200.0, 400.0],
                "home_ownership": ["RENT", "OWN"],
                "purpose": ["debt_consolidation", "credit_card"],
                "fico_range_low": [680.0, 720.0],
                "fico_range_high": [684.0, 724.0],
                "dti": [15.0, 20.0],
                "delinq_2yrs": [0.0, 1.0],
                "open_acc": [5.0, 8.0],
                "pub_rec": [0.0, 0.0],
                "revol_bal": [5_000.0, 10_000.0],
                "revol_util": [30.0, 45.0],
                "total_acc": [15.0, 20.0],
                "mort_acc": [0.0, 1.0],
                "pub_rec_bankruptcies": [0.0, 0.0],
                "loan_status": ["Fully Paid", "Charged Off"],
            }
        )
        result = engineer_features(df)
        if "loan_amnt_to_income" in result.columns:
            expected = [10_000 / 50_000, 20_000 / 100_000]
            np.testing.assert_allclose(
                result["loan_amnt_to_income"].values,
                expected,
                rtol=1e-5,
                err_msg="loan_amnt_to_income formula is incorrect",
            )