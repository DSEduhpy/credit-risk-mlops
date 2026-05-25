"""
Configuration package for the Credit Risk MLOps pipeline.

Environment resolution
----------------------
The active configuration is selected via the ``CREDIT_RISK_ENV`` environment
variable (default: ``dev``).  All application code should import ``settings``
from this module rather than instantiating environment-specific configs
directly::

    from src.config import settings

    cost = settings.business.default_cost
    model_dir = settings.paths.models

Backward compatibility
----------------------
All constants that existed in the original ``src/config.py`` flat module
are re-exported here so that existing import paths continue to work without
any changes to callers::

    # This still works after the migration to the config package:
    from src.config import DEFAULT_COST, REVENUE_PER_APPROVAL, PROJECT_ROOT
"""

from __future__ import annotations

import os
from typing import Union

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
    PROJECT_ROOT,
)

# ---------------------------------------------------------------------------
# Environment resolution
# ---------------------------------------------------------------------------
_ENV = os.getenv("CREDIT_RISK_ENV", "dev").lower()


def _resolve_settings(env: str) -> BaseConfig:
    """Instantiate the correct config class for the given environment tag."""
    if env == "prod":
        from src.config.prod import ProdConfig
        return ProdConfig()
    if env == "test":
        from src.config.test import TestConfig
        return TestConfig()
    # Default: dev (also covers any unrecognised value to avoid silent failures
    # in unknown environments — dev is the safest fallback)
    from src.config.dev import DevConfig
    return DevConfig()


settings: BaseConfig = _resolve_settings(_ENV)

# ---------------------------------------------------------------------------
# Backward-compatible re-exports
# These names replicate what src/config.py likely exposed so that no
# existing import in src/ or tests/ breaks.
# ---------------------------------------------------------------------------
DEFAULT_COST: int = settings.business.default_cost
REVENUE_PER_APPROVAL: int = settings.business.revenue_per_approval
DEFAULT_THRESHOLD: float = settings.business.default_threshold

RANDOM_STATE: int = settings.models.random_state
TEST_SIZE: float = settings.models.test_size
BENCHMARK_MODELS: list = settings.models.benchmark_models

MLFLOW_TRACKING_URI: str = settings.mlflow.tracking_uri
MLFLOW_EXPERIMENT_NAME: str = settings.mlflow.experiment_name

DATA_RAW_PATH: str = str(settings.paths.data_raw)
DATA_PROCESSED_PATH: str = str(settings.paths.data_processed)
DATA_FEATURES_PATH: str = str(settings.paths.data_features)
MODELS_PATH: str = str(settings.paths.models)
REPORTS_PATH: str = str(settings.paths.reports)

# Legacy aliases expected by older modules/tests
RAW_DATA_PATH: str = DATA_RAW_PATH
CLEAN_DATA_PATH: str = DATA_PROCESSED_PATH
FEATURE_DATA_PATH: str = DATA_FEATURES_PATH
FEATURES_PATH: str = DATA_FEATURES_PATH
MODEL_PATH: str = MODELS_PATH

TARGET_COLUMN: str = settings.validation.target_column

__all__ = [
    # Primary interface
    "settings",
    # Config sub-classes (for type hints and direct construction in tests)
    "BaseConfig",
    "PathConfig",
    "BusinessConfig",
    "MLflowConfig",
    "ModelConfig",
    "ValidationConfig",
    "MonitoringConfig",
    "APIConfig",
    "LoggingConfig",
    # Backward-compatible flat constants
    "PROJECT_ROOT",
    "DEFAULT_COST",
    "REVENUE_PER_APPROVAL",
    "DEFAULT_THRESHOLD",
    "RANDOM_STATE",
    "TEST_SIZE",
    "BENCHMARK_MODELS",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "DATA_RAW_PATH",
    "DATA_PROCESSED_PATH",
    "DATA_FEATURES_PATH",
    "MODELS_PATH",
    "REPORTS_PATH",
    "TARGET_COLUMN",
    "RAW_DATA_PATH",
    "CLEAN_DATA_PATH",
    "FEATURE_DATA_PATH",
    "MODEL_PATH",
    "FEATURES_PATH",
]