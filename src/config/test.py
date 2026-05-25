"""
Test environment configuration.

Designed for pytest runs: deterministic seeds, in-memory logging,
relaxed thresholds, isolated MLflow experiment to avoid polluting
the main experiment tracking database.

Activate by setting::

    CREDIT_RISK_ENV=test

or via conftest.py::

    import os
    os.environ["CREDIT_RISK_ENV"] = "test"
"""

from __future__ import annotations

from pathlib import Path

from src.config.base import (
    APIConfig,
    BaseConfig,
    BusinessConfig,
    LoggingConfig,
    MLflowConfig,
    ModelConfig,
    MonitoringConfig,
    PathConfig,
    ValidationConfig,
)

# ---------------------------------------------------------------------------
# Isolated test paths — tests must never read/write production data dirs
# ---------------------------------------------------------------------------
_TEST_ROOT = Path(__file__).resolve().parent.parent.parent / "tests" / "_fixtures"


class _TestPathConfig(PathConfig):
    data_raw: Path = _TEST_ROOT / "data" / "raw"
    data_processed: Path = _TEST_ROOT / "data" / "processed"
    data_features: Path = _TEST_ROOT / "data" / "features"
    models: Path = _TEST_ROOT / "models"
    reports: Path = _TEST_ROOT / "reports"
    reports_figures: Path = _TEST_ROOT / "reports" / "figures"
    reports_drift: Path = _TEST_ROOT / "reports" / "drift"
    logs: Path = _TEST_ROOT / "logs"


class _TestLoggingConfig(LoggingConfig):
    level: str = "WARNING"   # Suppress noise during test runs
    json_format: bool = False
    log_to_file: bool = False


class _TestMLflowConfig(MLflowConfig):
    tracking_uri: str = "file://" + str(_TEST_ROOT / "mlruns")
    experiment_name: str = "credit-risk-test"


class _TestModelConfig(ModelConfig):
    random_state: int = 42
    test_size: float = 0.3
    cv_folds: int = 2   # Faster in CI


class _TestMonitoringConfig(MonitoringConfig):
    min_drift_sample: int = 10  # Small fixtures are acceptable in tests


class _TestValidationConfig(ValidationConfig):
    min_row_count: int = 10   # Tests use small synthetic datasets


class _TestAPIConfig(APIConfig):
    port: int = 8001   # Avoid conflicts with running dev server


class TestConfig(BaseConfig):
    """
    Configuration for pytest test suite.

    Ensures tests are:
    - Isolated (separate paths from dev/prod)
    - Deterministic (fixed random_state)
    - Fast (reduced cv_folds, relaxed thresholds)
    - Non-destructive (no real MLflow / file writes unless explicitly tested)
    """

    environment: str = "test"

    paths: _TestPathConfig = _TestPathConfig()
    logging: _TestLoggingConfig = _TestLoggingConfig()
    mlflow: _TestMLflowConfig = _TestMLflowConfig()
    models: _TestModelConfig = _TestModelConfig()
    monitoring: _TestMonitoringConfig = _TestMonitoringConfig()
    validation: _TestValidationConfig = _TestValidationConfig()
    api: _TestAPIConfig = _TestAPIConfig()

    enable_drift_monitoring: bool = False
    enable_explainability: bool = False
    enable_great_expectations: bool = False