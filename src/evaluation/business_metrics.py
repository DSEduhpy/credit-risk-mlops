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