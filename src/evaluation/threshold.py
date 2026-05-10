import numpy as np

from sklearn.metrics import precision_recall_curve


def optimize_threshold(
    y_true,
    y_proba,
    custo_inadimplente=10000,
    lucro_cliente=1000,
):
    """
    Otimiza threshold baseado em resultado financeiro.

    Args:
        y_true:
            Valores reais

        y_proba:
            Probabilidades previstas pelo modelo

        custo_inadimplente:
            Prejuízo por falso negativo

        lucro_cliente:
            Lucro por cliente aprovado corretamente

    Returns:
        tuple:
            (
                best_threshold,
                best_resultado
            )
    """

    _, _, thresholds = precision_recall_curve(
        y_true,
        y_proba,
    )

    best_threshold = 0.5
    best_resultado = -np.inf

    for t in thresholds:

        y_pred = (y_proba >= t).astype(int)

        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        prejuizo = fn * custo_inadimplente
        lucro = tn * lucro_cliente

        resultado = lucro - prejuizo

        if resultado > best_resultado:
            best_resultado = resultado
            best_threshold = t

    return best_threshold, best_resultado