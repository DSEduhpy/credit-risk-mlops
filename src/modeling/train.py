"""Compatibilidade para o módulo antigo de treinamento.

Este arquivo reexporta a pipeline refatorada em ``src.modeling.pipeline``
afim de preservar imports e comportamentos existentes sem duplicar lógica.
"""

from src.modeling.pipeline import (
    DatasetSplit,
    PipelineRunner,
    log_to_mlflow,
    print_leaderboard,
    train,
    train_single_model,
)

__all__ = [
    "DatasetSplit",
    "PipelineRunner",
    "log_to_mlflow",
    "print_leaderboard",
    "train",
    "train_single_model",
]
