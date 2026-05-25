"""
Base configuration for the Credit Risk MLOps pipeline.

All environment-agnostic defaults live here. Environment-specific
configs (dev, test, prod) inherit from this class and override only
what differs. Do NOT import environment-specific modules here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Project root — resolved from this file's location so it works regardless
# of the working directory from which the pipeline is invoked.
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT: Path = _THIS_FILE.parent.parent.parent  # credit-risk-mlops/


# ---------------------------------------------------------------------------
# Directory layout — mirrors the current repository structure exactly.
# ---------------------------------------------------------------------------
class PathConfig:
    """Centralised path definitions. All paths are absolute."""

    root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    data_features: Path = PROJECT_ROOT / "data" / "features"
    models: Path = PROJECT_ROOT / "models"
    reports: Path = PROJECT_ROOT / "reports"
    reports_figures: Path = PROJECT_ROOT / "reports" / "figures"
    reports_drift: Path = PROJECT_ROOT / "reports" / "drift"
    mlruns: Path = PROJECT_ROOT / "mlruns"
    great_expectations: Path = PROJECT_ROOT / "great_expectations"
    logs: Path = PROJECT_ROOT / "logs"

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create all output directories if they do not already exist."""
        for attr, value in vars(cls).items():
            if isinstance(value, Path) and attr != "root":
                value.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Business economics — used by src/evaluation/business_metrics.py
# These are the core assumptions that drive profit optimisation.
# ---------------------------------------------------------------------------
class BusinessConfig:
    """Financial parameters for credit risk business logic."""

    default_cost: int = 10_000        # Cost of approving a customer who defaults
    revenue_per_approval: int = 1_000  # Revenue from a correctly approved customer
    default_threshold: float = 0.5    # Starting probability threshold (tuned later)


# ---------------------------------------------------------------------------
# MLflow tracking configuration
# ---------------------------------------------------------------------------
class MLflowConfig:
    """MLflow experiment tracking parameters."""

    tracking_uri: str = str(PROJECT_ROOT / "mlruns")
    experiment_name: str = "credit-risk-benchmark"
    artifact_location: Optional[str] = None  # None = default MLflow behaviour
    registered_model_name: str = "credit-risk-champion"

    # Stage labels used in the Model Registry workflow
    stage_staging: str = "Staging"
    stage_production: str = "Production"
    stage_archived: str = "Archived"


# ---------------------------------------------------------------------------
# Model benchmark — list of model keys used across train.py and evaluation
# ---------------------------------------------------------------------------
class ModelConfig:
    """Model identifiers and shared hyperparameter defaults."""

    benchmark_models: List[str] = [
        "logistic",
        "xgboost",
        "lightgbm",
        "catboost",
    ]

    # Random state applied wherever reproducibility is required
    random_state: int = 42

    # Cross-validation folds for threshold tuning
    cv_folds: int = 5

    # Test split ratio (used in src/modeling/data.py)
    test_size: float = 0.2

    # Model artifact file names — must match models/ directory
    artifact_names: Dict[str, str] = {
        "logistic": "logistic.pkl",
        "xgboost": "xgboost.pkl",
        "lightgbm": "lightgbm.pkl",
        "catboost": "catboost.pkl",
    }


# ---------------------------------------------------------------------------
# Data validation thresholds
# ---------------------------------------------------------------------------
class ValidationConfig:
    """Thresholds for data quality and schema validation."""

    # Maximum fraction of missing values allowed per column
    max_missing_rate: float = 0.05

    # Minimum number of rows expected in any processed dataset
    min_row_count: int = 100

    # Expected target column name
    target_column: str = "default"

    # Columns that must always be present after feature engineering
    required_feature_columns: List[str] = [
        "loan_amnt",
        "int_rate",
        "installment",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "fico_range_low",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "mort_acc",
        "pub_rec_bankruptcies",
        "home_ownership_encoded",
        "purpose_encoded",
        "loan_amnt_to_income",
        "fico_avg",
    ]


# ---------------------------------------------------------------------------
# Drift monitoring thresholds
# ---------------------------------------------------------------------------
class MonitoringConfig:
    """Thresholds that trigger drift alerts."""

    # Population Stability Index thresholds (industry standard)
    psi_warning: float = 0.1
    psi_critical: float = 0.2

    # KS statistic threshold for feature drift
    ks_threshold: float = 0.1

    # Wasserstein distance threshold
    wasserstein_threshold: float = 0.05

    # Minimum sample size for reliable drift computation
    min_drift_sample: int = 200

    # Prediction drift: allowed shift in mean predicted probability
    prediction_drift_threshold: float = 0.05


# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
class APIConfig:
    """FastAPI serving configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    api_version: str = "v1"
    request_timeout_seconds: int = 30


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
class LoggingConfig:
    """Structured logging defaults."""

    level: str = "INFO"
    json_format: bool = True
    log_to_file: bool = False
    log_file: Path = PROJECT_ROOT / "logs" / "pipeline.log"
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5


# ---------------------------------------------------------------------------
# Top-level convenience accessor — import this in application code
# ---------------------------------------------------------------------------
class BaseConfig:
    """
    Aggregated base configuration object.

    Usage in application code::

        from src.config import settings
        print(settings.business.default_cost)
        print(settings.paths.models)
    """

    paths: PathConfig = PathConfig()
    business: BusinessConfig = BusinessConfig()
    mlflow: MLflowConfig = MLflowConfig()
    models: ModelConfig = ModelConfig()
    validation: ValidationConfig = ValidationConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()

    # Environment tag — overridden by environment-specific subclasses
    environment: str = "base"

    # Feature flags — disabled by default, enabled per environment
    enable_drift_monitoring: bool = False
    enable_explainability: bool = True
    enable_great_expectations: bool = False