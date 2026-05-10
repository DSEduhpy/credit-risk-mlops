from typing import Optional, Sequence

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def plot_roc_curve(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plota a curva ROC para classificação binária."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("Curva ROC")
    return ax


def plot_precision_recall_curve(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plota a curva de precisão-recall para classificação binária."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("Curva Precision-Recall")
    return ax


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plota a matriz de confusão para previsões binárias."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title("Matriz de Confusão")
    return ax
