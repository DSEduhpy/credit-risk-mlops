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


import numpy as np


def compute_business_profit(
    y_true,
    y_prob,
    threshold=0.5,
    default_cost=10000,
    revenue_per_approval=1000,
):
    """
    Calcula lucro líquido do modelo.
    """

    if len(y_true) == 0:
        raise ValueError("Input vazio")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold deve estar entre 0 e 1")

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    approved = y_prob < threshold

    good_customers = approved & (y_true == 0)
    bad_customers = approved & (y_true == 1)

    profit = (
        good_customers.sum() * revenue_per_approval
        - bad_customers.sum() * default_cost
    )

    return float(profit)


def compute_profit_by_threshold(
    y_true,
    y_prob,
    thresholds=None,
):
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.05)

    results = {}

    for threshold in thresholds:
        results[float(threshold)] = compute_business_profit(
            y_true,
            y_prob,
            threshold=threshold,
        )

    return results




def find_optimal_threshold(
    y_true,
    y_prob,
    thresholds=None,
):
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.05)

    best_threshold = 0.5
    best_profit = float("-inf")

    for threshold in thresholds:

        profit = compute_business_profit(
            y_true,
            y_prob,
            threshold=threshold,
        )

        if profit > best_profit:
            best_profit = profit
            best_threshold = threshold

    return float(best_threshold)