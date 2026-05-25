"""
Unit tests for src/modeling/data.py.

Tests cover:
- Train/test split produces correct proportions
- Stratified split preserves class distribution
- Reproducibility: same seed produces identical splits
- Feature matrix excludes the target column
- Index is reset correctly after splitting
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.data import split_data


class TestSplitData:
    """Tests for the data splitting utility."""

    def test_returns_four_objects(self, feature_dataframe: pd.DataFrame):
        """split_data must return (X_train, X_test, y_train, y_test)."""
        result = split_data(feature_dataframe)
        assert len(result) == 4, (
            f"split_data returned {len(result)} objects, expected 4"
        )

    def test_split_ratio_approximately_correct(self, feature_dataframe: pd.DataFrame):
        """Test set should be ~20% of total data (within 5% tolerance)."""
        X_train, X_test, y_train, y_test = split_data(feature_dataframe, test_size=0.2)
        total = len(X_train) + len(X_test)
        actual_test_ratio = len(X_test) / total
        assert abs(actual_test_ratio - 0.2) <= 0.05, (
            f"Test ratio {actual_test_ratio:.2f} deviates from 0.20 by more than 5%"
        )

    def test_no_overlap_between_train_and_test(self, feature_dataframe: pd.DataFrame):
        """Train and test sets must be disjoint."""
        X_train, X_test, _, _ = split_data(feature_dataframe)
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        overlap = train_idx & test_idx
        assert not overlap, f"Train/test index overlap: {len(overlap)} shared rows"

    def test_reproducible_with_same_seed(self, feature_dataframe: pd.DataFrame):
        """Two calls with the same random_state must produce identical splits."""
        X_train_a, X_test_a, _, _ = split_data(feature_dataframe, random_state=42)
        X_train_b, X_test_b, _, _ = split_data(feature_dataframe, random_state=42)
        pd.testing.assert_frame_equal(X_train_a, X_train_b)
        pd.testing.assert_frame_equal(X_test_a, X_test_b)

    def test_different_seeds_produce_different_splits(
        self, feature_dataframe: pd.DataFrame
    ):
        """Different random_state values must (almost certainly) differ."""
        X_train_a, _, _, _ = split_data(feature_dataframe, random_state=1)
        X_train_b, _, _, _ = split_data(feature_dataframe, random_state=99)
        # Index order should differ; if they happen to be equal the dataset is tiny
        if len(feature_dataframe) > 20:
            assert not X_train_a.index.equals(X_train_b.index), (
                "Different seeds produced identical splits — likely a bug"
            )

    def test_target_not_in_feature_matrix(self, feature_dataframe: pd.DataFrame):
        """X_train and X_test must not contain the target column."""
        X_train, X_test, y_train, y_test = split_data(feature_dataframe)
        assert "default" not in X_train.columns, "Target 'default' found in X_train"
        assert "default" not in X_test.columns, "Target 'default' found in X_test"

    def test_target_shape_matches_feature_shape(self, feature_dataframe: pd.DataFrame):
        """y_train length must match X_train row count and vice versa for test."""
        X_train, X_test, y_train, y_test = split_data(feature_dataframe)
        assert len(X_train) == len(y_train), "X_train and y_train length mismatch"
        assert len(X_test) == len(y_test), "X_test and y_test length mismatch"

    def test_class_distribution_preserved(self, feature_dataframe: pd.DataFrame):
        """
        Stratified split should keep the default rate within ±10% of the
        original distribution in both train and test sets.
        """
        original_rate = feature_dataframe["default"].mean()
        X_train, X_test, y_train, y_test = split_data(
            feature_dataframe, stratify=True
        )
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - original_rate) <= 0.10, (
            f"Train default rate {train_rate:.2f} deviates from {original_rate:.2f}"
        )
        assert abs(test_rate - original_rate) <= 0.10, (
            f"Test default rate {test_rate:.2f} deviates from {original_rate:.2f}"
        )

    def test_feature_dtypes_preserved(self, feature_dataframe: pd.DataFrame):
        """Dtypes must not be coerced by the splitting operation."""
        X_train, X_test, _, _ = split_data(feature_dataframe)
        original_dtypes = feature_dataframe.drop(columns=["default"]).dtypes
        for col in X_train.columns:
            assert X_train[col].dtype == original_dtypes[col], (
                f"Dtype of '{col}' changed from {original_dtypes[col]} "
                f"to {X_train[col].dtype} after split"
            )