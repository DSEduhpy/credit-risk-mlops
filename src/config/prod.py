"""
Production environment configuration.

All monitoring and validation features are enabled. Logging is JSON-structured
for ingestion by CloudWatch / Datadog / ELK. No hot-reload, multi-worker API.

Activate by setting::

    CREDIT_RISK_ENV=prod

Never import this module directly in application code — always go through
``src.config.settings`` which resolves the environment automatically.
"""

from __future__ import annotations

import os

from src.config.base import (
    APIConfig,
    BaseConfig,
    LoggingConfig,
    MLflowConfig,
    MonitoringConfig,
)


class _ProdLoggingConfig(LoggingConfig):
    level: str = "INFO"
    json_format: bool = True    # Machine-parseable for log aggregation
    log_to_file: bool = True    # Written to logs/pipeline.log
    backup_count: int = 10


class _ProdAPIConfig(APIConfig):
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", "8000"))
    workers: int = int(os.getenv("API_WORKERS", "4"))
    reload: bool = False
    log_level: str = "info"
    request_timeout_seconds: int = 10


class _ProdMLflowConfig(MLflowConfig):
    # In production the tracking URI comes from the environment to support
    # remote MLflow servers (RDS-backed, S3 artifact store, etc.)
    tracking_uri: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://mlflow:5000",  # Docker Compose service name default
    )
    experiment_name: str = "credit-risk-benchmark"


class _ProdMonitoringConfig(MonitoringConfig):
    # Industry-standard PSI thresholds — no relaxation in prod
    psi_warning: float = 0.1
    psi_critical: float = 0.2
    ks_threshold: float = 0.1
    min_drift_sample: int = 500


class ProdConfig(BaseConfig):
    """
    Configuration for the production environment.

    Expectations:
    - CREDIT_RISK_ENV=prod is set in the container/deployment
    - MLFLOW_TRACKING_URI points to the remote MLflow server
    - PORT and API_WORKERS may be overridden via environment variables
    """

    environment: str = "prod"

    logging: _ProdLoggingConfig = _ProdLoggingConfig()
    api: _ProdAPIConfig = _ProdAPIConfig()
    mlflow: _ProdMLflowConfig = _ProdMLflowConfig()
    monitoring: _ProdMonitoringConfig = _ProdMonitoringConfig()

    enable_drift_monitoring: bool = True
    enable_explainability: bool = True
    enable_great_expectations: bool = True