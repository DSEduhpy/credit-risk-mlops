"""
Structured logging for the Credit Risk MLOps pipeline.

Features
--------
- JSON-formatted log records (configurable per environment)
- Correlation ID support for tracing requests across modules
- Full exception tracebacks in structured fields
- Pipeline-stage context injection
- Inference request logging with latency tracking
- Drift event logging with severity levels
- Rotating file handler (optional, enabled in prod)

Public interface (backward-compatible)
---------------------------------------
All existing call sites using ``get_logger(__name__)`` continue to work
without modification. The returned logger is a standard ``logging.Logger``
so all existing log calls (``logger.info``, ``logger.error``, etc.) are
unaffected.

Usage examples
--------------
Basic usage (backward-compatible)::

    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")

With correlation ID (new capability)::

    from src.logger import get_logger, set_correlation_id
    set_correlation_id("run-abc123")
    logger = get_logger(__name__)
    logger.info("Processing batch")   # ID injected automatically

Pipeline stage context::

    from src.logger import get_logger, pipeline_logger
    log = pipeline_logger("feature_engineering")
    log.info("Computing loan-to-income ratio")

Inference logging::

    from src.logger import get_logger, log_inference
    log_inference(request_id="req-001", model="xgboost",
                  prediction=0.72, latency_ms=12.4)

Drift alert::

    from src.logger import log_drift_alert
    log_drift_alert(feature="annual_inc", psi=0.23, threshold=0.2)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Correlation ID context variable
# Each async context / thread can carry its own correlation ID without
# interference. Falls back to "no-correlation-id" when not set.
# ---------------------------------------------------------------------------
_correlation_id_var: ContextVar[str] = ContextVar(
    "_correlation_id", default="no-correlation-id"
)


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set the active correlation ID for the current execution context.

    If ``correlation_id`` is None a UUID4 is generated automatically.
    Returns the ID that was set so callers can log or propagate it.
    """
    cid = correlation_id or str(uuid.uuid4())
    _correlation_id_var.set(cid)
    return cid


def get_correlation_id() -> str:
    """Return the active correlation ID (or the 'no-correlation-id' sentinel)."""
    return _correlation_id_var.get()


# ---------------------------------------------------------------------------
# JSON log formatter
# ---------------------------------------------------------------------------
class _JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects suitable for ingestion
    by CloudWatch Logs, Datadog, or any ELK-compatible log aggregator.

    Every record includes:
    - timestamp (ISO-8601, UTC)
    - level
    - logger name
    - message
    - correlation_id
    - module / function / line for traceability
    - exc_info as a structured ``exception`` field (not a raw string)
    - Any extra fields passed via ``extra={}``
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Inject any extra fields the caller passed via extra={}
        _reserved = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in _reserved and not key.startswith("_"):
                payload[key] = value

        # Structured exception field
        if record.exc_info and record.exc_info[0] is not None:
            exc_type, exc_value, exc_tb = record.exc_info
            payload["exception"] = {
                "type": exc_type.__name__ if exc_type else "Unknown",
                "message": str(exc_value),
                "traceback": traceback.format_exception(exc_type, exc_value, exc_tb),
            }

        return json.dumps(payload, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Human-readable formatter (used in dev / test environments)
# ---------------------------------------------------------------------------
class _HumanFormatter(logging.Formatter):
    """
    Coloured, human-readable formatter for terminal output.
    Includes the correlation ID in brackets when it has been set.
    """

    _COLOURS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    _RESET = "\033[0m"
    _BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = self._COLOURS.get(record.levelname, "")
        cid = get_correlation_id()
        cid_str = f" [{cid[:8]}]" if cid != "no-correlation-id" else ""
        prefix = (
            f"{colour}{self._BOLD}{record.levelname:<8}{self._RESET}"
            f" {datetime.fromtimestamp(record.created, tz=timezone.utc).strftime('%H:%M:%S')}"
            f"{cid_str}"
            f" {record.name} —"
        )
        message = f"{prefix} {record.getMessage()}"

        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return message


# ---------------------------------------------------------------------------
# Logger registry — prevents duplicate handlers on repeated get_logger calls
# ---------------------------------------------------------------------------
_configured_loggers: set[str] = set()


def _configure_root_logger() -> None:
    """
    Configure the root logger once at module import time.

    Reads environment settings lazily to avoid circular imports with
    src.config at module level.
    """
    # Lazy import to avoid circular dependency: logger <- config <- logger
    try:
        from src.config import settings
        use_json = settings.logging.json_format
        level = getattr(logging, settings.logging.level.upper(), logging.INFO)
        log_to_file = settings.logging.log_to_file
        log_file = settings.logging.log_file
        max_bytes = settings.logging.max_bytes
        backup_count = settings.logging.backup_count
    except Exception:
        # Fallback when config is not yet available (e.g. during bootstrapping)
        use_json = False
        level = logging.INFO
        log_to_file = False
        log_file = Path("logs/pipeline.log")
        max_bytes = 10 * 1024 * 1024
        backup_count = 5

    root = logging.getLogger()

    # Avoid re-configuring if already set up
    if root.handlers:
        return

    root.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter: logging.Formatter = (
        _JSONFormatter() if use_json else _HumanFormatter()
    )
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Rotating file handler (production)
    if log_to_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(_JSONFormatter())  # Always JSON in files
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)


# Run once at import
_configure_root_logger()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    This is the primary public interface and is backward-compatible with
    all existing call sites. Callers do not need to configure handlers —
    the root logger is already configured at module import.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A standard ``logging.Logger`` instance.
    """
    return logging.getLogger(name)


def pipeline_logger(stage: str) -> logging.Logger:
    """
    Return a logger pre-tagged with a pipeline stage name.

    Produces log records under the ``pipeline.<stage>`` namespace so
    pipeline events are easily filtered in aggregated logs.

    Args:
        stage: DVC pipeline stage name (e.g. ``"feature_engineering"``).

    Returns:
        Logger under ``pipeline.<stage>``.
    """
    logger = logging.getLogger(f"pipeline.{stage}")
    return logger


def log_inference(
    request_id: str,
    model: str,
    prediction: float,
    latency_ms: float,
    features: Optional[Dict[str, Any]] = None,
    threshold: float = 0.5,
) -> None:
    """
    Emit a structured inference log record.

    Writes to the ``inference`` logger namespace so inference events can
    be routed to a dedicated log stream or monitored independently.

    Args:
        request_id: Unique request identifier (from X-Request-ID header).
        model: Model name used for scoring (e.g. ``"xgboost"``).
        prediction: Raw probability score output by the model.
        latency_ms: End-to-end inference latency in milliseconds.
        features: Optional dict of input feature values (for audit logging).
        threshold: Decision threshold applied to produce the binary decision.
    """
    logger = logging.getLogger("inference")
    decision = "approved" if prediction < threshold else "denied"
    logger.info(
        "Inference completed",
        extra={
            "request_id": request_id,
            "model": model,
            "prediction_score": round(prediction, 6),
            "decision": decision,
            "threshold": threshold,
            "latency_ms": round(latency_ms, 3),
            "feature_count": len(features) if features else 0,
        },
    )


def log_drift_alert(
    feature: str,
    metric: str,
    value: float,
    threshold: float,
    severity: str = "WARNING",
) -> None:
    """
    Emit a structured drift alert log record.

    Args:
        feature: Feature name where drift was detected.
        metric: Drift metric name (e.g. ``"PSI"``, ``"KS"``, ``"wasserstein"``).
        value: Computed drift metric value.
        threshold: Threshold that was exceeded.
        severity: ``"WARNING"`` or ``"CRITICAL"`` (maps to log level).
    """
    logger = logging.getLogger("monitoring.drift")
    level = logging.CRITICAL if severity.upper() == "CRITICAL" else logging.WARNING
    logger.log(
        level,
        f"Drift detected on feature '{feature}'",
        extra={
            "feature": feature,
            "drift_metric": metric,
            "drift_value": round(value, 6),
            "threshold": threshold,
            "severity": severity.upper(),
            "alert_type": "feature_drift",
        },
    )


def log_pipeline_stage(
    stage: str,
    status: str,
    duration_seconds: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit a structured pipeline stage completion record.

    Args:
        stage: DVC stage name.
        status: ``"started"``, ``"completed"``, or ``"failed"``.
        duration_seconds: Wall-clock duration of the stage, if known.
        metadata: Arbitrary key-value metadata to attach to the record.
    """
    logger = logging.getLogger(f"pipeline.{stage}")
    extra: Dict[str, Any] = {
        "stage": stage,
        "status": status,
    }
    if duration_seconds is not None:
        extra["duration_seconds"] = round(duration_seconds, 3)
    if metadata:
        extra.update(metadata)

    level = logging.ERROR if status == "failed" else logging.INFO
    logger.log(level, f"Pipeline stage '{stage}' {status}", extra=extra)


# ---------------------------------------------------------------------------
# Context manager for timing pipeline stages
# ---------------------------------------------------------------------------
class StageTimer:
    """
    Context manager that logs pipeline stage start/end with timing.

    Usage::

        from src.logger import StageTimer
        with StageTimer("feature_engineering"):
            run_feature_engineering()
    """

    def __init__(self, stage: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.stage = stage
        self.metadata = metadata or {}
        self._start: float = 0.0

    def __enter__(self) -> "StageTimer":
        self._start = time.monotonic()
        log_pipeline_stage(self.stage, "started", metadata=self.metadata)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration = time.monotonic() - self._start
        status = "failed" if exc_type is not None else "completed"
        log_pipeline_stage(
            self.stage,
            status,
            duration_seconds=duration,
            metadata=self.metadata,
        )