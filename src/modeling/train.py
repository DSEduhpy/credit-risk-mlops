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

import time
from pathlib import Path
from typing import Dict, List

import joblib
import mlflow
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
)
from src.logger import get_logger

from src.modeling.data import (
    load_features,
    split_data,
)

from src.modeling.models import (
    build_catboost_model,
    build_lightgbm_model,
    build_logistic_model,
    build_xgboost_model,
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

# Dicionário de modelos para benchmark
MODELS = {
    "logistic": build_logistic_model,
    "xgboost": build_xgboost_model,
    "lightgbm": build_lightgbm_model,
    "catboost": build_catboost_model,
}


def create_pipeline(model) -> Pipeline:
    """Cria pipeline com scaler e modelo."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])


def train_single_model(
    model_name: str,
    model_func,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """Treina um único modelo e retorna métricas."""
    logger.info(f"Iniciando treinamento do modelo: {model_name}")

    start_time = time.perf_counter()

    # Construir pipeline
    model = create_pipeline(model_func())

    # Treinar
    model.fit(X_train, y_train)

    # Gerar probabilidades
    y_proba = model.predict_proba(X_test)[:, 1]

    # Otimizar threshold
    best_threshold, best_resultado = optimize_threshold(
        y_true=y_test,
        y_proba=y_proba,
        custo_inadimplente=10000,
        lucro_cliente=1000,
    )

    y_pred = (y_proba >= best_threshold).astype(int)

    # Métricas técnicas
    metrics_dict = compute_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    # Métricas de negócio
    business_metrics = simulate_business_metrics(
        y_true=y_test,
        y_pred=y_pred,
        custo_inadimplente=10000,
        lucro_cliente=1000,
    )

    training_time = time.perf_counter() - start_time

    logger.info(
        f"Modelo {model_name} finalizado",
        extra={
            "model": model_name,
            "training_time": training_time,
            "best_threshold": best_threshold,
            "resultado": business_metrics["resultado"],
        }
    )

    return {
        "model_name": model_name,
        "model": model,
        "metrics": metrics_dict,
        "business_metrics": business_metrics,
        "best_threshold": best_threshold,
        "training_time": training_time,
    }


def log_to_mlflow(model_name: str, result: Dict) -> None:
    """Registra experimento no MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Parâmetros
        mlflow.log_param("model", model_name)
        mlflow.log_param("threshold", float(result["best_threshold"]))
        mlflow.log_param("training_time", float(result["training_time"]))

        # Métricas técnicas
        for key, value in result["metrics"].items():
            mlflow.log_metric(key, float(value))

        # Métricas negócio
        for key, value in result["business_metrics"].items():
            mlflow.log_metric(key, float(value))

        # Salvar modelo
        model_path = PROJECT_ROOT.parent / "models" / f"{model_name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result["model"], model_path)

        mlflow.log_artifact(str(model_path), artifact_path="model")


def print_leaderboard(results: List[Dict]) -> None:
    """Imprime leaderboard final no terminal."""
    print("\n===== BENCHMARK FINAL =====")
    print(f"{'Modelo':<12} | {'AUC':<6} | {'Precision':<10} | {'Recall':<8} | {'Resultado':<10}")
    print("-" * 60)

    for result in results:
        auc = result["metrics"].get("auc", 0)
        precision = result["metrics"].get("precision", 0)
        recall = result["metrics"].get("recall", 0)
        resultado = result["business_metrics"].get("resultado", 0)

        print(f"{result['model_name']:<12} | {auc:<6.4f} | {precision:<10.4f} | {recall:<8.4f} | {resultado:<10.0f}")

    print("=" * 60)


def train() -> None:
    """
    Executa benchmark multi-modelo.
    """
    logger.info("Iniciando benchmark multi-modelo")

    # Configurar MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("credit_risk_benchmark")

    # Carregar dados
    data = load_features()
    X_train, X_test, y_train, y_test = split_data(data)

    results = []

    # Loop de benchmark
    for model_name, model_func in MODELS.items():
        result = train_single_model(
            model_name=model_name,
            model_func=model_func,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Logar no MLflow
        log_to_mlflow(model_name, result)

        results.append(result)

    # Leaderboard final
    print_leaderboard(results)

    logger.info("Benchmark concluído", extra={"num_models": len(results)})


if __name__ == "__main__":
    train()