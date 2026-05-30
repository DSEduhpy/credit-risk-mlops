from typing import Dict, Sequence

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calcula métricas de classificação a partir das probabilidades.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold deve estar entre 0 e 1")

    y_pred = (np.array(y_prob) >= threshold).astype(int)

    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
    }