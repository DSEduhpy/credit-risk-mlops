from __future__ import annotations

from src.monitoring.monitoring import ModelMonitor


def test_monitoring_records_and_alerts(tmp_path) -> None:
    """O monitor deve registrar snapshots e detectar alertas de degradação."""
    monitor = ModelMonitor(history_path=tmp_path / "history.json")
    monitor.record_snapshot(10, 0.80, 0.20, 120.0, 0.01)
    monitor.record_snapshot(20, 0.55, 0.30, 180.0, 0.20)

    alerts = monitor.evaluate_alerts(threshold=0.05)

    assert len(alerts) >= 1
    assert monitor.get_history()[0].inference_count == 10
