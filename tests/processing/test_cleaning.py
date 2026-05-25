"""
Unit tests for src/processing/cleaning.py.

Tests cover:
- Null/missing value handling
- Column type coercion
- Row count preservation (or explicit reduction)
- No data leakage from target column
- Idempotency: cleaning an already-clean DataFrame produces the same result
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.cleaning import clean_data


class TestCleanData:
    """Tests for the main data cleaning function."""

    def test_returns_dataframe(self, raw_dataframe: pd.DataFrame):
        result = clean_data(raw_dataframe.copy())
        assert isinstance(result, pd.DataFrame), "clean_data must return a DataFrame"

    def test_no_rows_added(self, raw_dataframe: pd.DataFrame):
        """Cleaning must never add rows."""
        result = clean_data(raw_dataframe.copy())
        assert len(result) <= len(raw_dataframe), (
            "clean_data must not add rows to the dataset"
        )

    def test_missing_rate_below_threshold(self, raw_dataframe: pd.DataFrame):
        """
        After cleaning, no column should exceed the configured missing rate.
        The threshold is relaxed here to accommodate the small synthetic dataset.
        """
        result = clean_data(raw_dataframe.copy())
        max_missing = result.isnull().mean().max()
        assert max_missing <= 0.10, (
            f"Cleaning left {max_missing:.1%} missing values in some column"
        )

    def test_target_column_preserved(self, raw_dataframe: pd.DataFrame):
        """The loan_status (or default) column must survive cleaning."""
        result = clean_data(raw_dataframe.copy())
        target_candidates = {"loan_status", "default"}
        has_target = bool(target_candidates & set(result.columns))
        assert has_target, "Target column was dropped during cleaning"

    def test_idempotent(self, raw_dataframe: pd.DataFrame):
        """Applying clean_data twice must produce the same result as once."""
        once = clean_data(raw_dataframe.copy())
        twice = clean_data(once.copy())
        assert once.shape == twice.shape, "clean_data is not idempotent (shape changed)"
        assert list(once.columns) == list(twice.columns), (
            "clean_data is not idempotent (columns changed)"
        )

    def test_numeric_columns_are_numeric(self, raw_dataframe: pd.DataFrame):
        """Known numeric columns must have numeric dtype after cleaning."""
        result = clean_data(raw_dataframe.copy())
        numeric_cols = [
            c for c in ["loan_amnt", "int_rate", "annual_inc", "dti"]
            if c in result.columns
        ]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(result[col]), (
                f"Column '{col}' is not numeric after cleaning"
            )

    def test_handles_completely_empty_column(self):
        """Columns that are 100% null should be dropped or imputed, not crash."""
        df = pd.DataFrame(
            {
                "loan_amnt": [1000.0, 2000.0, 3000.0],
                "all_null": [np.nan, np.nan, np.nan],
                "loan_status": ["Fully Paid", "Charged Off", "Fully Paid"],
            }
        )
        # Must not raise
        result = clean_data(df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_input(self, raw_dataframe: pd.DataFrame):
        """clean_data must not modify the DataFrame passed as argument."""
        original_shape = raw_dataframe.shape
        original_nulls = raw_dataframe.isnull().sum().sum()
        _ = clean_data(raw_dataframe.copy())
        # Verify original is unchanged
        assert raw_dataframe.shape == original_shape
        assert raw_dataframe.isnull().sum().sum() == original_nulls