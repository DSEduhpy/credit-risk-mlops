"""
Unit tests for src/config/ package.

Tests cover:
- TestConfig is resolved when CREDIT_RISK_ENV=test
- All backward-compatible re-exports are present
- Sub-config values have correct types
- PathConfig returns Path objects
- BusinessConfig values match project assumptions
- Different environments produce different settings
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


class TestConfigResolution:
    """Tests for environment-based configuration resolution."""

    def test_test_env_resolves_test_config(self):
        """CREDIT_RISK_ENV=test (set in conftest.py) must resolve TestConfig."""
        from src.config import settings
        assert settings.environment == "test", (
            f"Expected environment='test', got '{settings.environment}'"
        )

    def test_settings_is_not_none(self):
        from src.config import settings
        assert settings is not None

    def test_dev_config_resolves_correctly(self):
        """Explicitly instantiating DevConfig must produce environment='dev'."""
        from src.config.dev import DevConfig
        dev = DevConfig()
        assert dev.environment == "dev"

    def test_prod_config_resolves_correctly(self):
        """Explicitly instantiating ProdConfig must produce environment='prod'."""
        from src.config.prod import ProdConfig
        prod = ProdConfig()
        assert prod.environment == "prod"

    def test_test_config_resolves_correctly(self):
        from src.config.test import TestConfig
        test = TestConfig()
        assert test.environment == "test"


class TestBackwardCompatibleExports:
    """Tests that the flat-module import interface still works."""

    def test_default_cost_exported(self):
        from src.config import DEFAULT_COST
        assert isinstance(DEFAULT_COST, int)
        assert DEFAULT_COST > 0

    def test_revenue_per_approval_exported(self):
        from src.config import REVENUE_PER_APPROVAL
        assert isinstance(REVENUE_PER_APPROVAL, int)
        assert REVENUE_PER_APPROVAL > 0

    def test_random_state_exported(self):
        from src.config import RANDOM_STATE
        assert isinstance(RANDOM_STATE, int)

    def test_benchmark_models_exported(self):
        from src.config import BENCHMARK_MODELS
        assert isinstance(BENCHMARK_MODELS, list)
        assert len(BENCHMARK_MODELS) >= 1

    def test_mlflow_tracking_uri_exported(self):
        from src.config import MLFLOW_TRACKING_URI
        assert isinstance(MLFLOW_TRACKING_URI, str)
        assert len(MLFLOW_TRACKING_URI) > 0

    def test_target_column_exported(self):
        from src.config import TARGET_COLUMN
        assert isinstance(TARGET_COLUMN, str)
        assert TARGET_COLUMN == "default"

    def test_project_root_exported(self):
        from src.config import PROJECT_ROOT
        assert isinstance(PROJECT_ROOT, Path)
        assert PROJECT_ROOT.exists(), "PROJECT_ROOT must point to an existing directory"


class TestBusinessConfig:
    """Tests for BusinessConfig values."""

    def test_default_cost_is_positive(self):
        from src.config import settings
        assert settings.business.default_cost > 0

    def test_revenue_is_positive(self):
        from src.config import settings
        assert settings.business.revenue_per_approval > 0

    def test_default_cost_exceeds_revenue(self):
        """Default cost must exceed revenue to make the optimisation meaningful."""
        from src.config import settings
        assert settings.business.default_cost > settings.business.revenue_per_approval, (
            "default_cost must exceed revenue_per_approval for meaningful optimisation"
        )

    def test_default_threshold_in_unit_interval(self):
        from src.config import settings
        t = settings.business.default_threshold
        assert 0.0 < t < 1.0, f"default_threshold {t} must be in (0, 1)"


class TestPathConfig:
    """Tests for PathConfig."""

    def test_all_paths_are_path_objects(self):
        from src.config import settings
        paths = settings.paths
        for attr in ("data_raw", "data_processed", "data_features", "models", "reports"):
            value = getattr(paths, attr)
            assert isinstance(value, Path), (
                f"paths.{attr} is {type(value)}, expected Path"
            )

    def test_test_paths_use_test_fixtures_dir(self):
        """In the test environment, paths must point inside the test fixture area."""
        from src.config import settings
        # Test paths must be separated from production paths
        assert "test" in str(settings.paths.models).lower() or \
               "_fixtures" in str(settings.paths.models), (
            f"Test model path does not look like a test path: {settings.paths.models}"
        )


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_benchmark_models_contains_expected_entries(self):
        from src.config import settings
        models = settings.models.benchmark_models
        expected = {"logistic", "xgboost", "lightgbm", "catboost"}
        assert expected <= set(models), (
            f"Missing benchmark models: {expected - set(models)}"
        )

    def test_test_size_in_valid_range(self):
        from src.config import settings
        ts = settings.models.test_size
        assert 0.1 <= ts <= 0.5, f"test_size {ts} is outside [0.1, 0.5]"

    def test_random_state_is_non_negative(self):
        from src.config import settings
        assert settings.models.random_state >= 0


class TestMonitoringConfig:
    """Tests for MonitoringConfig threshold ordering."""

    def test_psi_warning_below_critical(self):
        from src.config import settings
        m = settings.monitoring
        assert m.psi_warning < m.psi_critical, (
            f"PSI warning threshold ({m.psi_warning}) must be < critical ({m.psi_critical})"
        )

    def test_min_drift_sample_positive(self):
        from src.config import settings
        assert settings.monitoring.min_drift_sample > 0


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_level_is_valid(self):
        import logging
        from src.config import settings
        level_str = settings.logging.level.upper()
        assert hasattr(logging, level_str), (
            f"Invalid logging level: {level_str}"
        )

    def test_test_env_disables_file_logging(self):
        """File logging must be off in the test environment to avoid I/O noise."""
        from src.config import settings
        assert settings.logging.log_to_file is False, (
            "Test environment must not write to log files"
        )