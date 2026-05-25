"""
Development environment configuration.

Inherits all defaults from BaseConfig and overrides values appropriate
for local development: verbose logging, reload enabled, monitoring disabled.
"""

from __future__ import annotations

from src.config.base import (
    APIConfig,
    BaseConfig,
    LoggingConfig,
    MLflowConfig,
    MonitoringConfig,
)


class _DevLoggingConfig(LoggingConfig):
    level: str = "DEBUG"
    json_format: bool = False   # Human-readable in dev terminals
    log_to_file: bool = False


class _DevAPIConfig(APIConfig):
    reload: bool = True
    workers: int = 1
    log_level: str = "debug"


class _DevMLflowConfig(MLflowConfig):
    experiment_name: str = "credit-risk-benchmark-dev"


class _DevMonitoringConfig(MonitoringConfig):
    # Relax thresholds locally to avoid noise during development
    psi_critical: float = 0.3
    min_drift_sample: int = 50


class DevConfig(BaseConfig):
    """
    Configuration for local development.

    Activate by setting the environment variable::

        CREDIT_RISK_ENV=dev

    Or by importing directly::

        from src.config.dev import DevConfig
        settings = DevConfig()
    """

    environment: str = "dev"

    logging: _DevLoggingConfig = _DevLoggingConfig()
    api: _DevAPIConfig = _DevAPIConfig()
    mlflow: _DevMLflowConfig = _DevMLflowConfig()
    monitoring: _DevMonitoringConfig = _DevMonitoringConfig()

    enable_drift_monitoring: bool = False
    enable_explainability: bool = True
    enable_great_expectations: bool = False