from typing import Dict, Optional, Sequence

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Computa métricas de avaliação de classificação binária."""
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": (sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)),
    }

    if y_proba is not None:
        metrics["auc"] = roc_auc_score(y_true, y_proba)

    return metrics

