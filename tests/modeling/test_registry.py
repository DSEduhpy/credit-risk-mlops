from __future__ import annotations

from pathlib import Path

from src.modeling.registry import ModelRegistry


def test_register_and_promote_model(tmp_path: Path) -> None:
    """O registro deve persistir metadados e permitir promoção para produção."""
    registry = ModelRegistry(registry_path=tmp_path / "registry.json")

    registry.register(
        model_name="candidate",
        model_path="/tmp/candidate.pkl",
        stage="Development",
        metadata={"score": 0.82},
    )
    registry.register(
        model_name="champion",
        model_path="/tmp/champion.pkl",
        stage="Development",
        metadata={"score": 0.90},
    )

    promoted = registry.promote("champion", target_stage="Production")

    assert promoted["stage"] == "Production"
    active_model = registry.get_active_model()
    assert active_model is not None
    assert active_model["name"] == "champion"
