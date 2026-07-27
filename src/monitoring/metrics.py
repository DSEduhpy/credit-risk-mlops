"""Métricas observáveis para os componentes de inferência e treinamento."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelMetrics:
    """Resumo simples de métricas de inferência e modelo."""

    inference_count: int = 0
    training_count: int = 0
    last_probability: float = 0.0
    last_prediction: int = 0

    def record_inference(self, probability: float, prediction: int) -> None:
        """Registra uma inferência e atualiza o estado mais recente."""
        self.inference_count += 1
        self.last_probability = probability
        self.last_prediction = prediction

    def record_training(self) -> None:
        """Registra uma execução de treinamento."""
        self.training_count += 1

    def to_prometheus(self) -> dict[str, Any]:
        """Retorna o estado em formato de métricas simples."""
        return {
            "inference_count": self.inference_count,
            "training_count": self.training_count,
            "last_probability": self.last_probability,
            "last_prediction": self.last_prediction,
        }
