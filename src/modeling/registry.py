"""
Registro simples de modelos para o pipeline de risco de crédito.

Este módulo oferece uma camada leve de registry para acompanhar modelos
candidatos, campeões e modelos em produção sem depender de um sistema
externo como MLflow. A implementação é compatível com o fluxo atual e
permite evolução futura para um registry profissional.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Gerencia metadados de modelos e sua promoção entre estágios profissionais."""

    def __init__(self, registry_path: str | Path | None = None) -> None:
        self.registry_path = Path(registry_path or "models/registry.json")
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, Any] = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        """Carrega o estado atual do registry, se existir."""
        if not self.registry_path.exists():
            return {"models": []}

        try:
            with self.registry_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, OSError) as error:
            logger.warning("Falha ao carregar registry: %s", error)

        return {"models": []}

    def _save_registry(self) -> None:
        """Persiste o estado do registry em disco."""
        with self.registry_path.open("w", encoding="utf-8") as handle:
            json.dump(self._registry, handle, indent=2, ensure_ascii=False)

    def register(
        self,
        model_name: str,
        model_path: str | Path,
        stage: str = "Development",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Registra um novo modelo com metadados e histórico."""
        entry = {
            "name": model_name,
            "path": str(model_path),
            "stage": stage,
            "version": self._next_version(),
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "history": [],
        }
        self._registry.setdefault("models", []).append(entry)
        self._save_registry()
        logger.info(
            "Modelo registrado", extra={"model_name": model_name, "stage": stage}
        )
        return deepcopy(entry)

    def promote(
        self, model_name: str, target_stage: str = "Production"
    ) -> dict[str, Any]:
        """Atualiza o estágio de um modelo e registra o histórico de promoção."""
        for entry in self._registry.setdefault("models", []):
            if entry.get("name") != model_name:
                continue

            previous_stage = entry.get("stage")
            entry["stage"] = target_stage
            entry.setdefault("history", []).append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "from_stage": previous_stage,
                    "to_stage": target_stage,
                }
            )
            self._save_registry()
            logger.info(
                "Modelo promovido",
                extra={"model_name": model_name, "target_stage": target_stage},
            )
            return deepcopy(entry)

        raise KeyError(f"Modelo '{model_name}' não encontrado no registry")

    def get_active_model(self) -> dict[str, Any] | None:
        """Retorna o modelo ativo em Production; se não existir, tenta Champion."""
        for entry in reversed(self._registry.setdefault("models", [])):
            if entry.get("stage") == "Production":
                return deepcopy(entry)

        for entry in reversed(self._registry.setdefault("models", [])):
            if entry.get("stage") == "Champion":
                return deepcopy(entry)

        return None

    def rollback(self, model_name: str) -> dict[str, Any]:
        """Reverte o estágio do modelo para o estágio anterior do histórico."""
        for entry in self._registry.setdefault("models", []):
            if entry.get("name") != model_name:
                continue

            history = entry.setdefault("history", [])
            if len(history) < 2:
                raise ValueError(
                    f"Modelo '{model_name}' não possui histórico suficiente para rollback"
                )

            previous = history[-2]
            entry["stage"] = previous.get("to_stage", "Development")
            history.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "from_stage": entry.get("stage"),
                    "to_stage": previous.get("from_stage", "Development"),
                }
            )
            self._save_registry()
            logger.info(
                "Rollback aplicado",
                extra={"model_name": model_name, "target_stage": entry["stage"]},
            )
            return deepcopy(entry)

        raise KeyError(f"Modelo '{model_name}' não encontrado no registry")

    def list_models(self) -> list[dict[str, Any]]:
        """Retorna a lista completa de modelos registrados."""
        return [deepcopy(entry) for entry in self._registry.setdefault("models", [])]

    def _next_version(self) -> int:
        """Calcula a próxima versão numérica do registro."""
        return len(self._registry.setdefault("models", [])) + 1
