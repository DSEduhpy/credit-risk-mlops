"""Serviços de inferência e carregamento de modelos para a API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow

from src.config import MLFLOW_TRACKING_URI, MODEL_PATH
from src.logger import get_logger
from src.modeling.registry import ModelRegistry

logger = get_logger(__name__)


class ModelService:
    """Responsável por carregar e manter o modelo ativo para inferência."""

    def __init__(self, registry_path: str | Path | None = None) -> None:
        self.registry = ModelRegistry(registry_path or "models/registry.json")
        self.model: Any = None

    def load_model(self) -> Any:
        """Carrega o modelo campeão a partir do registry, do disco ou do MLflow."""
        active_model = self.registry.get_active_model()
        candidate_path = Path(active_model["path"]) if active_model else None

        if candidate_path and candidate_path.exists():
            logger.info(
                "Carregando modelo do registry",
                extra={"model_path": str(candidate_path)},
            )
            self.model = joblib.load(candidate_path)
            return self.model

        fallback_path = Path(MODEL_PATH)
        if fallback_path.exists():
            logger.info(
                "Carregando modelo local de fallback",
                extra={"model_path": str(fallback_path)},
            )
            self.model = joblib.load(fallback_path)
            return self.model

        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_candidates = list(
                Path(MLFLOW_TRACKING_URI).rglob("*/artifacts/model")
            )
            if model_candidates:
                latest_model = sorted(
                    model_candidates,
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )[0]
                logger.info(
                    "Carregando modelo do MLflow",
                    extra={"mlflow_model_path": str(latest_model)},
                )
                self.model = mlflow.sklearn.load_model(str(latest_model))
                return self.model
        except Exception as error:
            logger.warning("Falha ao carregar modelo do MLflow: %s", error)

        raise FileNotFoundError(
            "Modelo não encontrado no registry, em models/ nem no MLflow local."
        )
