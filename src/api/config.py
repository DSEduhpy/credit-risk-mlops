"""Configurações da API FastAPI para o serviço de risco de crédito."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import settings


@dataclass(frozen=True)
class APISettings:
    """Parâmetros operacionais da interface HTTP."""

    title: str = "Credit Risk MLOps API"
    version: str = "1.0.0"
    service_name: str = "credit-risk-mlops-api"
    host: str = settings.api.host
    port: int = settings.api.port
    log_level: str = settings.api.log_level
    timeout_seconds: int = settings.api.request_timeout_seconds


api_settings = APISettings()
