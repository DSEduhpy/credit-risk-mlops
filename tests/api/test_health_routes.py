from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_health_ready_and_version_endpoints() -> None:
    """As rotas de saúde e versão devem responder com sucesso."""
    with (
        patch("joblib.load", return_value=object()),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run"),
    ):
        from src.api.app import app

        client = TestClient(app, raise_server_exceptions=True)
        response_health = client.get("/health")
        response_ready = client.get("/ready")
        response_version = client.get("/version")

        assert response_health.status_code == 200
        assert response_ready.status_code == 200
        assert response_version.status_code == 200
