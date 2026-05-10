"""
Pipeline de treinamento e benchmark multi-modelo para risco de crédito.

Responsabilidades:
- Carregar dados
- Separar treino/teste
- Benchmark entre múltiplos modelos
- Otimizar threshold para cada modelo
- Calcular métricas técnicas e financeiras
- Registrar experimentos no MLflow
- Persistir modelos treinados
"""

# pylint: disable=invalid-name

import time

from dataclasses import dataclass
from typing import Any

import joblib
import mlflow
import pandas as pd

from src.config import (
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
)

from src.logger import get_logger

from src.modeling.data import (
    load_features,
    split_data,
)

from src.modeling.models.logistic import (
    build_logistic_model,
)

from src.modeling.models.xgboost import (
    build_xgboost_model,
)

from src.modeling.models.lightgbm import (
    build_lightgbm_model,
)

from src.modeling.models.catboost import (
    build_catboost_model,
)

from src.evaluation.metrics import (
    compute_metrics,
)

from src.evaluation.threshold import (
    optimize_threshold,
)

from src.evaluation.business_metrics import (
    simulate_business_metrics,
)

logger = get_logger(__name__)

# ==========================================================
# Modelos disponíveis para benchmark
# ==========================================================
MODELS = {
    "logistic": build_logistic_model,
    "xgboost": build_xgboost_model,
    "lightgbm": build_lightgbm_model,
    "catboost": build_catboost_model,
}


@dataclass
class DatasetSplit:
    """
    Estrutura contendo divisão de treino e teste.
    """

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def train_single_model(
    model_name: str,
    model_func,
    data: DatasetSplit,
) -> dict[str, Any]:
    """
    Treina um único modelo e retorna métricas.
    """

    logger.info(
        "Iniciando treinamento do modelo: %s",
        model_name,
    )

    start_time = time.perf_counter()

    # ==========================================================
    # Construção do modelo
    # ==========================================================
    model = model_func()

    # ==========================================================
    # Treinamento
    # ==========================================================
    model.fit(data.x_train, data.y_train)

    # ==========================================================
    # Probabilidades
    # ==========================================================
    y_proba = model.predict_proba(data.x_test)[:, 1]
    # ==========================================================
    # Otimização de threshold
    # ==========================================================
    best_threshold = optimize_threshold(
        y_true=data.y_test,
        y_proba=y_proba,
        custo_inadimplente=10000,
        lucro_cliente=1000,
    )

    y_pred = (y_proba >= best_threshold).astype(int)

    # ==========================================================
    # Métricas técnicas
    # ==========================================================
    metrics_dict = compute_metrics(
        y_true=data.y_test,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    # ==========================================================
    # Métricas financeiras
    # ==========================================================
    business_metrics = simulate_business_metrics(
        y_true=data.y_test,
        y_pred=y_pred,
        custo_inadimplente=10000,
        lucro_cliente=1000,
    )

    training_time = time.perf_counter() - start_time

    logger.info(
        "Modelo %s finalizado",
        model_name,
        extra={
            "model": model_name,
            "training_time": training_time,
            "best_threshold": best_threshold,
            "resultado": business_metrics["resultado"],
        },
    )

    return {
        "model_name": model_name,
        "model": model,
        "metrics": metrics_dict,
        "business_metrics": business_metrics,
        "best_threshold": best_threshold,
        "training_time": training_time,
    }


def log_to_mlflow(
    model_name: str,
    result: dict[str, Any],
) -> None:
    """
    Registra experimento no MLflow.
    """

    with mlflow.start_run(run_name=model_name):

        # ==========================================================
        # Parâmetros
        # ==========================================================
        mlflow.log_param("model", model_name)

        mlflow.log_param(
            "threshold",
            float(result["best_threshold"]),
        )

        mlflow.log_param(
            "training_time",
            float(result["training_time"]),
        )

        # ==========================================================
        # Métricas técnicas
        # ==========================================================
        for key, value in result["metrics"].items():
            mlflow.log_metric(key, float(value))

        # ==========================================================
        # Métricas financeiras
        # ==========================================================
        for key, value in result["business_metrics"].items():
            mlflow.log_metric(key, float(value))

        # ==========================================================
        # Persistência do modelo
        # ==========================================================
        model_path = (
            PROJECT_ROOT.parent
            / "models"
            / f"{model_name}.pkl"
        )

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            result["model"],
            model_path,
        )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="model",
        )


def print_leaderboard(
    results: list[dict[str, Any]],
) -> None:
    """
    Imprime leaderboard final.
    """

    print("\n===== BENCHMARK FINAL =====")

    print(
        f"{'Modelo':<12} | "
        f"{'AUC':<6} | "
        f"{'Precision':<10} | "
        f"{'Recall':<8} | "
        f"{'Resultado':<10}"
    )

    print("-" * 60)

    for result in results:

        auc = result["metrics"].get("auc", 0)
        precision = result["metrics"].get("precision", 0)
        recall = result["metrics"].get("recall", 0)

        resultado = result[
            "business_metrics"
        ].get("resultado", 0)

        print(
            f"{result['model_name']:<12} | "
            f"{auc:<6.4f} | "
            f"{precision:<10.4f} | "
            f"{recall:<8.4f} | "
            f"{resultado:<10.0f}"
        )

    print("=" * 60)


def train() -> None:
    """
    Executa benchmark multi-modelo.
    """

    logger.info(
        "Iniciando benchmark multi-modelo"
    )

    # ==========================================================
    # Configuração MLflow
    # ==========================================================
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        "credit_risk_benchmark"
    )

    # ==========================================================
    # Dados
    # ==========================================================
    data = load_features()

    x_train, x_test, y_train, y_test = split_data(data)

    dataset = DatasetSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )

    results = []

    # ==========================================================
    # Benchmark
    # ==========================================================
    for model_name, model_func in MODELS.items():

        result = train_single_model(
            model_name=model_name,
            model_func=model_func,
            data=dataset,
    )

        # MLflow
        log_to_mlflow(
            model_name,
            result,
        )

        results.append(result)

    # ==========================================================
    # Leaderboard
    # ==========================================================
    print_leaderboard(results)

    logger.info(
        "Benchmark concluído",
        extra={
            "num_models": len(results),
        },
    )


if __name__ == "__main__":
    train()
