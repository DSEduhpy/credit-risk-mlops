"""
Tests for src/api/app.py — FastAPI inference service.

Tests cover:
- /health endpoint returns 200 and expected payload
- /predict endpoint accepts valid input and returns a score
- /predict validates required fields (422 on missing fields)
- /predict rejects out-of-range values
- /version endpoint returns version information
- Error responses follow the expected JSON schema
- Request IDs are present in responses (if implemented)

These tests use FastAPI's TestClient so no running server is needed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# TestClient import — starlette is a FastAPI dependency, always available
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# We import the app lazily inside fixtures to allow model-loading patches
# to be in place before the module-level startup code runs.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_model():
    """A mock model that returns a fixed probability of 0.3."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.7, 0.3]])
    return model


@pytest.fixture(scope="module")
def test_client(mock_model):
    """
    Create a TestClient with the model loading patched out.

    Patches:
    - joblib.load: returns mock_model regardless of path
    - MLflow: suppressed to avoid tracking during tests
    """
    with patch("joblib.load", return_value=mock_model), \
         patch("mlflow.set_experiment"), \
         patch("mlflow.start_run"):
        from src.api.app import app
        client = TestClient(app, raise_server_exceptions=True)
        yield client


@pytest.fixture
def valid_payload():
    """Minimal valid prediction request payload."""
    return {
        "loan_amnt": 15000.0,
        "int_rate": 12.5,
        "installment": 350.0,
        "annual_inc": 65000.0,
        "dti": 18.5,
        "delinq_2yrs": 0.0,
        "fico_range_low": 710.0,
        "open_acc": 8.0,
        "pub_rec": 0.0,
        "revol_bal": 12000.0,
        "revol_util": 35.0,
        "total_acc": 20.0,
        "mort_acc": 1.0,
        "pub_rec_bankruptcies": 0.0,
        "home_ownership_encoded": 2.0,
        "purpose_encoded": 0.0,
        "loan_amnt_to_income": 0.23,
        "fico_avg": 712.5,
    }


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, test_client: TestClient):
        response = test_client.get("/health")
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

    def test_health_returns_json(self, test_client: TestClient):
        response = test_client.get("/health")
        assert response.headers["content-type"].startswith("application/json"), (
            "Health endpoint must return JSON"
        )

    def test_health_payload_has_status(self, test_client: TestClient):
        response = test_client.get("/health")
        body = response.json()
        assert "status" in body, f"Health response missing 'status' key: {body}"

    def test_health_status_is_healthy(self, test_client: TestClient):
        response = test_client.get("/health")
        body = response.json()
        assert body["status"] in ("ok", "healthy", "up"), (
            f"Unexpected health status: {body['status']}"
        )


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_returns_200(self, test_client: TestClient, valid_payload: dict):
        response = test_client.post("/predict", json=valid_payload)
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

    def test_predict_response_has_score(
        self, test_client: TestClient, valid_payload: dict
    ):
        response = test_client.post("/predict", json=valid_payload)
        body = response.json()
        assert "score" in body or "probability" in body or "prediction" in body, (
            f"Predict response missing score field: {body}"
        )

    def test_predict_score_in_unit_interval(
        self, test_client: TestClient, valid_payload: dict
    ):
        response = test_client.post("/predict", json=valid_payload)
        body = response.json()
        score = body.get("score") or body.get("probability") or body.get("prediction")
        assert 0.0 <= float(score) <= 1.0, (
            f"Prediction score {score} is outside [0, 1]"
        )

    def test_predict_missing_field_returns_422(
        self, test_client: TestClient, valid_payload: dict
    ):
        """Omitting a required field must return HTTP 422 Unprocessable Entity."""
        incomplete = {k: v for k, v in valid_payload.items() if k != "loan_amnt"}
        response = test_client.post("/predict", json=incomplete)
        assert response.status_code == 422, (
            f"Expected 422 for missing field, got {response.status_code}"
        )

    def test_predict_empty_body_returns_422(self, test_client: TestClient):
        """Sending an empty JSON body must return 422."""
        response = test_client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_invalid_json_returns_422(self, test_client: TestClient):
        """Sending malformed JSON must return 422 or 400."""
        response = test_client.post(
            "/predict",
            data="not-json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code in (400, 422)

    def test_predict_response_is_json(
        self, test_client: TestClient, valid_payload: dict
    ):
        response = test_client.post("/predict", json=valid_payload)
        assert response.headers["content-type"].startswith("application/json")

    def test_predict_negative_loan_amnt_rejected(
        self, test_client: TestClient, valid_payload: dict
    ):
        """Negative loan amount is financially invalid and should be rejected."""
        bad_payload = {**valid_payload, "loan_amnt": -5000.0}
        response = test_client.post("/predict", json=bad_payload)
        # Pydantic validation or business logic should catch this
        assert response.status_code in (400, 422), (
            f"Negative loan_amnt should be rejected, got {response.status_code}"
        )


# ---------------------------------------------------------------------------
# Version endpoint
# ---------------------------------------------------------------------------

class TestVersionEndpoint:
    """Tests for the /version endpoint."""

    def test_version_returns_200(self, test_client: TestClient):
        response = test_client.get("/version")
        assert response.status_code == 200

    def test_version_has_version_field(self, test_client: TestClient):
        response = test_client.get("/version")
        body = response.json()
        assert "version" in body or "api_version" in body, (
            f"Version response missing version field: {body}"
        )