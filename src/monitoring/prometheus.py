"""Exposição simples de métricas em formato Prometheus."""

from __future__ import annotations

from src.monitoring.metrics import ModelMetrics


class PrometheusMetrics:
    """Gera um payload textual compatível com Prometheus."""

    def __init__(self, metrics: ModelMetrics | None = None) -> None:
        self.metrics = metrics or ModelMetrics()

    def render(self) -> str:
        """Retorna um texto com métricas no formato Prometheus."""
        values = self.metrics.to_prometheus()
        lines = [
            "# HELP credit_risk_inference_count Total de inferências processadas",
            "# TYPE credit_risk_inference_count counter",
            f"credit_risk_inference_count {values['inference_count']}",
            "# HELP credit_risk_training_count Total de execuções de treinamento",
            "# TYPE credit_risk_training_count counter",
            f"credit_risk_training_count {values['training_count']}",
            "# HELP credit_risk_last_probability Última probabilidade registrada",
            "# TYPE credit_risk_last_probability gauge",
            f"credit_risk_last_probability {values['last_probability']}",
            "# HELP credit_risk_last_prediction Última decisão registrada",
            "# TYPE credit_risk_last_prediction gauge",
            f"credit_risk_last_prediction {values['last_prediction']}",
        ]
        return "\n".join(lines)
