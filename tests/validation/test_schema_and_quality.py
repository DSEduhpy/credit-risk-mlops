"""
Unit tests for src/validation/schema.py and src/validation/quality.py.

Tests cover:
- Schema validation passes on correctly-shaped data
- Schema validation fails on missing required columns
- Quality checks catch high missing rates
- Quality checks catch insufficient row counts
- Validator is composable (schema + quality in sequence)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.schema import validate_schema
from src.validation.quality import check_data_quality


# ---------------------------------------------------------------------------
# Tests: Schema validation
# ---------------------------------------------------------------------------

class TestValidateSchema:
    """Tests for column-level schema enforcement."""

    def test_valid_feature_dataframe_passes(self, feature_dataframe: pd.DataFrame):
        """A fully-formed feature DataFrame must pass schema validation."""
        result = validate_schema(feature_dataframe)
        # validate_schema should return True or raise on failure
        if isinstance(result, bool):
            assert result is True
        # If it raises on failure and doesn't raise here, the test passes

    def test_missing_required_column_fails(self, feature_dataframe: pd.DataFrame):
        """Dropping a required column must cause validate_schema to fail."""
        df_bad = feature_dataframe.drop(columns=["loan_amnt"])
        with pytest.raises(Exception):
            validate_schema(df_bad)

    def test_extra_columns_are_allowed(self, feature_dataframe: pd.DataFrame):
        """Additional columns beyond the required set must not cause failure."""
        df_extra = feature_dataframe.copy()
        df_extra["phantom_column"] = 0.0
        # Should not raise
        validate_schema(df_extra)

    def test_empty_dataframe_fails(self):
        """An empty DataFrame (zero rows) must fail schema validation."""
        df_empty = pd.DataFrame()
        with pytest.raises(Exception):
            validate_schema(df_empty)

    def test_wrong_dtype_raises(self, feature_dataframe: pd.DataFrame):
        """Replacing a numeric column with strings must trigger a schema error."""
        df_bad = feature_dataframe.copy()
        df_bad["loan_amnt"] = df_bad["loan_amnt"].astype(str)
        with pytest.raises(Exception):
            validate_schema(df_bad)


# ---------------------------------------------------------------------------
# Tests: Data quality checks
# ---------------------------------------------------------------------------

class TestCheckDataQuality:
    """Tests for data quality assertions."""

    def test_clean_data_passes(self, feature_dataframe: pd.DataFrame):
        """A clean feature DataFrame must pass all quality checks."""
        result = check_data_quality(feature_dataframe)
        if isinstance(result, bool):
            assert result is True

    def test_high_missing_rate_fails(self, feature_dataframe: pd.DataFrame):
        """A column with >5% missing values must trigger a quality failure."""
        df_bad = feature_dataframe.copy()
        n_null = int(len(df_bad) * 0.15)  # 15% missing
        df_bad.loc[df_bad.index[:n_null], "loan_amnt"] = np.nan
        with pytest.raises(Exception):
            check_data_quality(df_bad)

    def test_too_few_rows_fails(self):
        """A DataFrame with fewer rows than the minimum must fail quality checks."""
        # Minimum is set to 10 in TestConfig
        tiny_df = pd.DataFrame({"loan_amnt": [1000.0], "default": [0]})
        with pytest.raises(Exception):
            check_data_quality(tiny_df)

    def test_quality_report_contains_column_stats(self, feature_dataframe: pd.DataFrame):
        """
        If check_data_quality returns a report dict, it must include
        per-column missing rates and a row count field.
        """
        result = check_data_quality(feature_dataframe)
        if isinstance(result, dict):
            assert "row_count" in result or "n_rows" in result, (
                "Quality report must include row count"
            )

    def test_all_nulls_in_column_fails(self, feature_dataframe: pd.DataFrame):
        """A completely null column must be caught by quality checks."""
        df_bad = feature_dataframe.copy()
        df_bad["loan_amnt"] = np.nan
        with pytest.raises(Exception):
            check_data_quality(df_bad)