from __future__ import annotations

from src.monitoring.metrics import ModelMetrics
from src.monitoring.prometheus import PrometheusMetrics


def test_metrics_record_and_render() -> None:
    """As métricas devem registrar inferências e gerar um payload textual."""
    metrics = ModelMetrics()
    metrics.record_inference(probability=0.82, prediction=1)
    metrics.record_training()

    rendered = PrometheusMetrics(metrics).render()

    assert "credit_risk_inference_count 1" in rendered
    assert "credit_risk_training_count 1" in rendered
    assert "credit_risk_last_probability 0.82" in rendered
