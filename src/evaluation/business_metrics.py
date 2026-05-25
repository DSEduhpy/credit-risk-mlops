import numpy as np


def simulate_business_metrics(
    y_true,
    y_pred,
    custo_inadimplente=10000,
    lucro_cliente=1000,
):
    """
    Simula impacto financeiro do modelo.

    Args:
        y_true:
            Valores reais

        y_pred:
            Predições binárias

        custo_inadimplente:
            Prejuízo por inadimplente aprovado

        lucro_cliente:
            Lucro por cliente saudável aprovado

    Returns:
        dict com métricas financeiras
    """

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    prejuizo = fn * custo_inadimplente
    lucro = tn * lucro_cliente
    resultado = lucro - prejuizo

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "lucro": float(lucro),
        "prejuizo": float(prejuizo),
        "resultado": float(resultado),
    }


def compute_business_profit(
    y_true,
    y_pred,
    custo_inadimplente=10000,
    lucro_cliente=1000,
):
    """
    Backward-compatible wrapper used by automated tests.
    """

    return simulate_business_metrics(
        y_true=y_true,
        y_pred=y_pred,
        custo_inadimplente=custo_inadimplente,
        lucro_cliente=lucro_cliente,
    )


def compute_profit_by_threshold(
    y_true,
    y_pred,
    custo_inadimplente=10000,
    lucro_cliente=1000,
):
    """
    Compatibility wrapper for threshold-based business evaluation.
    """

    return simulate_business_metrics(
        y_true=y_true,
        y_pred=y_pred,
        custo_inadimplente=custo_inadimplente,
        lucro_cliente=lucro_cliente,
    )




def find_optimal_threshold(
    y_true,
    y_prob,
    thresholds=None,
    custo_inadimplente=10000,
    lucro_cliente=1000,
):
    """
    Finds threshold with best financial outcome.
    Backward-compatible implementation for automated tests.
    """

    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)

    best_threshold = 0.5
    best_result = float("-inf")
    best_metrics = None

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        metrics = simulate_business_metrics(
            y_true=y_true,
            y_pred=y_pred,
            custo_inadimplente=custo_inadimplente,
            lucro_cliente=lucro_cliente,
        )

        resultado = metrics["resultado"]

        if resultado > best_result:
            best_result = resultado
            best_threshold = threshold
            best_metrics = metrics

    return {
        "best_threshold": float(best_threshold),
        "best_result": float(best_result),
        "metrics": best_metrics,
    }