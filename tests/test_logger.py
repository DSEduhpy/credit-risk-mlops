"""
Unit tests for src/logger.py.

Tests cover:
- get_logger returns a standard logging.Logger
- JSON formatter produces valid JSON with required keys
- Human formatter produces readable output
- Correlation ID is injected into log records
- set_correlation_id returns the set ID
- pipeline_logger uses the correct namespace
- log_inference emits a record with required extra fields
- log_drift_alert emits WARNING or CRITICAL depending on severity
- StageTimer context manager logs start and completion
- StageTimer logs failure on exception
"""

from __future__ import annotations

import json
import logging
import time
from io import StringIO
from unittest.mock import patch

import pytest

from src.logger import (
    StageTimer,
    _HumanFormatter,
    _JSONFormatter,
    get_correlation_id,
    get_logger,
    log_drift_alert,
    log_inference,
    log_pipeline_stage,
    pipeline_logger,
    set_correlation_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_logger(logger_name: str, formatter_cls=None):
    """
    Attach a StringIO handler to a logger and return (logger, stream, handler).
    Caller must remove the handler after use.
    """
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    fmt = formatter_cls() if formatter_cls else _JSONFormatter()
    handler.setFormatter(fmt)

    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger, stream, handler


# ---------------------------------------------------------------------------
# Tests: get_logger
# ---------------------------------------------------------------------------

class TestGetLogger:
    def test_returns_logger_instance(self):
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_matches(self):
        logger = get_logger("test.specific.name")
        assert logger.name == "test.specific.name"

    def test_repeated_calls_return_same_instance(self):
        a = get_logger("test.singleton")
        b = get_logger("test.singleton")
        assert a is b


# ---------------------------------------------------------------------------
# Tests: Correlation ID
# ---------------------------------------------------------------------------

class TestCorrelationId:
    def test_set_returns_the_id(self):
        cid = set_correlation_id("my-custom-id")
        assert cid == "my-custom-id"
        assert get_correlation_id() == "my-custom-id"

    def test_set_without_arg_generates_uuid(self):
        cid = set_correlation_id()
        assert len(cid) == 36, f"UUID4 should be 36 chars, got {len(cid)}: {cid}"
        assert get_correlation_id() == cid

    def test_correlation_id_in_json_output(self):
        set_correlation_id("json-test-id")
        logger, stream, handler = _capture_logger(
            "test.json.cid", _JSONFormatter
        )
        try:
            logger.info("test message")
            output = stream.getvalue().strip()
            record = json.loads(output)
            assert record.get("correlation_id") == "json-test-id"
        finally:
            logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Tests: JSON formatter
# ---------------------------------------------------------------------------

class TestJSONFormatter:
    def test_output_is_valid_json(self):
        logger, stream, handler = _capture_logger("test.json.format", _JSONFormatter)
        try:
            logger.info("hello from json formatter")
            output = stream.getvalue().strip()
            parsed = json.loads(output)
            assert isinstance(parsed, dict)
        finally:
            logger.removeHandler(handler)

    def test_required_keys_present(self):
        logger, stream, handler = _capture_logger("test.json.keys", _JSONFormatter)
        try:
            logger.info("key check")
            output = stream.getvalue().strip()
            record = json.loads(output)
            for key in ("timestamp", "level", "logger", "message", "correlation_id"):
                assert key in record, f"Key '{key}' missing from JSON log record"
        finally:
            logger.removeHandler(handler)

    def test_level_matches(self):
        logger, stream, handler = _capture_logger("test.json.level", _JSONFormatter)
        try:
            logger.warning("this is a warning")
            output = stream.getvalue().strip()
            record = json.loads(output)
            assert record["level"] == "WARNING"
        finally:
            logger.removeHandler(handler)

    def test_exception_is_structured(self):
        logger, stream, handler = _capture_logger("test.json.exc", _JSONFormatter)
        try:
            try:
                raise ValueError("synthetic error")
            except ValueError:
                logger.exception("caught an error")
            output = stream.getvalue().strip()
            record = json.loads(output)
            assert "exception" in record, "Exception field missing from JSON record"
            exc = record["exception"]
            assert "type" in exc
            assert exc["type"] == "ValueError"
            assert "message" in exc
            assert "traceback" in exc
        finally:
            logger.removeHandler(handler)

    def test_extra_fields_included(self):
        logger, stream, handler = _capture_logger("test.json.extra", _JSONFormatter)
        try:
            logger.info("with extras", extra={"request_id": "req-123", "model": "xgb"})
            output = stream.getvalue().strip()
            record = json.loads(output)
            assert record.get("request_id") == "req-123"
            assert record.get("model") == "xgb"
        finally:
            logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Tests: Human formatter
# ---------------------------------------------------------------------------

class TestHumanFormatter:
    def test_output_is_string(self):
        logger, stream, handler = _capture_logger(
            "test.human.format", _HumanFormatter
        )
        try:
            logger.info("human readable message")
            output = stream.getvalue()
            assert isinstance(output, str)
            assert len(output) > 0
        finally:
            logger.removeHandler(handler)

    def test_level_in_output(self):
        logger, stream, handler = _capture_logger(
            "test.human.level", _HumanFormatter
        )
        try:
            logger.error("an error occurred")
            output = stream.getvalue()
            assert "ERROR" in output
        finally:
            logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Tests: pipeline_logger
# ---------------------------------------------------------------------------

class TestPipelineLogger:
    def test_namespace_is_prefixed(self):
        log = pipeline_logger("feature_engineering")
        assert log.name == "pipeline.feature_engineering"

    def test_pipeline_logger_logs_without_error(self):
        log = pipeline_logger("test_stage")
        log.info("pipeline stage running")  # Must not raise


# ---------------------------------------------------------------------------
# Tests: log_inference
# ---------------------------------------------------------------------------

class TestLogInference:
    def test_emits_record(self):
        logger, stream, handler = _capture_logger("inference", _JSONFormatter)
        try:
            log_inference(
                request_id="req-001",
                model="lightgbm",
                prediction=0.65,
                latency_ms=11.2,
                threshold=0.5,
            )
            output = stream.getvalue().strip()
            assert output, "log_inference produced no log output"
            record = json.loads(output)
            assert record.get("request_id") == "req-001"
            assert record.get("model") == "lightgbm"
            assert "prediction_score" in record
            assert "latency_ms" in record
            assert "decision" in record
        finally:
            logger.removeHandler(handler)

    def test_decision_approved_when_below_threshold(self):
        logger, stream, handler = _capture_logger("inference", _JSONFormatter)
        try:
            log_inference(
                request_id="r1", model="xgboost",
                prediction=0.2, latency_ms=5.0, threshold=0.5
            )
            record = json.loads(stream.getvalue().strip())
            assert record["decision"] == "approved"
        finally:
            logger.removeHandler(handler)

    def test_decision_denied_when_above_threshold(self):
        logger, stream, handler = _capture_logger("inference", _JSONFormatter)
        try:
            log_inference(
                request_id="r2", model="xgboost",
                prediction=0.8, latency_ms=5.0, threshold=0.5
            )
            record = json.loads(stream.getvalue().strip())
            assert record["decision"] == "denied"
        finally:
            logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Tests: log_drift_alert
# ---------------------------------------------------------------------------

class TestLogDriftAlert:
    def test_warning_severity_maps_to_warning_level(self):
        logger, stream, handler = _capture_logger(
            "monitoring.drift", _JSONFormatter
        )
        try:
            log_drift_alert("annual_inc", "PSI", 0.15, 0.10, severity="WARNING")
            output = stream.getvalue().strip()
            record = json.loads(output)
            assert record["level"] == "WARNING"
        finally:
            logger.removeHandler(handler)

    def test_critical_severity_maps_to_critical_level(self):
        logger, stream, handler = _capture_logger(
            "monitoring.drift", _JSONFormatter
        )
        try:
            log_drift_alert("dti", "KS", 0.25, 0.10, severity="CRITICAL")
            output = stream.getvalue().strip()
            record = json.loads(output)
            assert record["level"] == "CRITICAL"
        finally:
            logger.removeHandler(handler)

    def test_record_contains_feature_name(self):
        logger, stream, handler = _capture_logger(
            "monitoring.drift", _JSONFormatter
        )
        try:
            log_drift_alert("revol_util", "wasserstein", 0.08, 0.05, "WARNING")
            record = json.loads(stream.getvalue().strip())
            assert record.get("feature") == "revol_util"
        finally:
            logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Tests: StageTimer
# ---------------------------------------------------------------------------

class TestStageTimer:
    def test_logs_started_and_completed(self):
        logger, stream, handler = _capture_logger(
            "pipeline.timer_test", _JSONFormatter
        )
        try:
            with StageTimer("timer_test"):
                pass
            lines = [l for l in stream.getvalue().strip().split("\n") if l]
            statuses = [json.loads(l).get("status") for l in lines]
            assert "started" in statuses, "StageTimer did not log 'started'"
            assert "completed" in statuses, "StageTimer did not log 'completed'"
        finally:
            logger.removeHandler(handler)

    def test_logs_failed_on_exception(self):
        logger, stream, handler = _capture_logger(
            "pipeline.fail_test", _JSONFormatter
        )
        try:
            with pytest.raises(RuntimeError):
                with StageTimer("fail_test"):
                    raise RuntimeError("deliberate failure")
            lines = [l for l in stream.getvalue().strip().split("\n") if l]
            statuses = [json.loads(l).get("status") for l in lines]
            assert "failed" in statuses, (
                "StageTimer did not log 'failed' on exception"
            )
        finally:
            logger.removeHandler(handler)

    def test_duration_is_recorded(self):
        logger, stream, handler = _capture_logger(
            "pipeline.duration_test", _JSONFormatter
        )
        try:
            with StageTimer("duration_test"):
                time.sleep(0.01)
            lines = [l for l in stream.getvalue().strip().split("\n") if l]
            completed = [
                json.loads(l) for l in lines
                if json.loads(l).get("status") == "completed"
            ]
            assert completed, "No 'completed' record found"
            assert "duration_seconds" in completed[0], (
                "StageTimer completion record missing duration_seconds"
            )
            assert completed[0]["duration_seconds"] >= 0.005
        finally:
            logger.removeHandler(handler)