"""Monitoramento profissional de modelo para drift, métricas e alertas."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MonitoringSnapshot:
    """Representa um snapshot de métricas de produção para monitoramento."""

    timestamp: str
    inference_count: int
    average_probability: float
    positive_rate: float
    avg_inference_time_ms: float
    error_rate: float


class ModelMonitor:
    """Orquestra coleta, persistência e alerta de degradação de modelo."""

    def __init__(self, history_path: str | Path | None = None) -> None:
        self.history_path = Path(
            history_path or "reports/monitoring_history.json")
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[MonitoringSnapshot] = self._load_history()

    def _load_history(self) -> list[MonitoringSnapshot]:
        """Carrega o histórico de snapshots, se existir."""
        if not self.history_path.exists():
            return []

        try:
            with self.history_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
                return [MonitoringSnapshot(**item) for item in payload]
        except (json.JSONDecodeError, OSError, TypeError) as error:
            logger.warning("Falha ao carregar histórico: %s", error)
            return []

    def _save_history(self) -> None:
        """Persiste o histórico de snapshots em disco."""
        with self.history_path.open("w", encoding="utf-8") as handle:
            json.dump([asdict(item)
                      for item in self._history], handle, indent=2)

    def record_snapshot(
        self,
        inference_count: int,
        average_probability: float,
        positive_rate: float,
        avg_inference_time_ms: float,
        error_rate: float,
    ) -> MonitoringSnapshot:
        """Registra um novo snapshot de observabilidade de produção."""
        snapshot = MonitoringSnapshot(
            timestamp=pd.Timestamp.utcnow().isoformat(),
            inference_count=inference_count,
            average_probability=average_probability,
            positive_rate=positive_rate,
            avg_inference_time_ms=avg_inference_time_ms,
            error_rate=error_rate,
        )
        self._history.append(snapshot)
        self._save_history()
        logger.info("Snapshot de monitoramento registrado",
                    extra=asdict(snapshot))
        return snapshot

    def get_history(self) -> list[MonitoringSnapshot]:
        """Retorna o histórico completo de snapshots."""
        return list(self._history)

    def evaluate_alerts(self, threshold: float = 0.15) -> list[dict[str, Any]]:
        """Identifica degradação significativa em relação ao último snapshot."""
        if len(self._history) < 2:
            return []

        current = self._history[-1]
        previous = self._history[-2]
        alerts: list[dict[str, Any]] = []

        if current.average_probability < previous.average_probability - threshold:
            alerts.append(
                {
                    "type": "probability_drop",
                    "message": "Média de probabilidade caiu significativamente",
                }
            )

        if current.error_rate > previous.error_rate + threshold:
            alerts.append(
                {"type": "error_rate_spike", "message": "Taxa de erro aumentou"}
            )

        return alerts
