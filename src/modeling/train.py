"""
Módulo de treinamento de modelo de risco de crédito.
"""

import joblib
import mlflow
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURES_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.logger import get_logger

logger = get_logger(__name__)


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {FEATURES_PATH}")
    return pd.read_parquet(FEATURES_PATH)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def train() -> None:
    logger.info("Iniciando treinamento do modelo")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("credit_risk")

    # =============================
    # Dados
    # =============================
    data = load_features()

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Coluna alvo '{TARGET_COLUMN}' não encontrada")

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # =============================
    # Modelo
    # =============================
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    # =============================
    # Predições
    # =============================
    y_proba = model.predict_proba(X_test)[:, 1]

    # =============================
    # Otimização de threshold (NEGÓCIO)
    # =============================
    custo_inadimplente = 10000
    lucro_cliente = 1000

    _, _, thresholds = precision_recall_curve(y_test, y_proba)

    best_resultado = -np.inf
    best_threshold = 0.5

    for t in thresholds:
        y_pred_temp = (y_proba >= t).astype(int)

        fn = ((y_pred_temp == 0) & (y_test == 1)).sum()
        tn = ((y_pred_temp == 0) & (y_test == 0)).sum()

        prejuizo_temp = fn * custo_inadimplente
        lucro_temp = tn * lucro_cliente
        resultado_temp = lucro_temp - prejuizo_temp

        if resultado_temp > best_resultado:
            best_resultado = resultado_temp
            best_threshold = t

    # Predição final
    y_pred = (y_proba >= best_threshold).astype(int)

    # =============================
    # Métricas
    # =============================
    metrics_dict = compute_metrics(y_test.values, y_pred, y_proba)

    # =============================
    # IMPACTO FINANCEIRO CORRETO
    # =============================
    tp = ((y_pred == 1) & (y_test == 1)).sum()
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    tn = ((y_pred == 0) & (y_test == 0)).sum()

    prejuizo = fn * custo_inadimplente
    lucro = tn * lucro_cliente
    resultado = lucro - prejuizo

    # =============================
    # Logs no terminal
    # =============================
    print("\n===== MODELO OTIMIZADO =====")
    print(f"Threshold: {best_threshold:.4f}")
    print(f"AUC: {metrics_dict['auc']:.4f}")
    print(f"Precision: {metrics_dict['precision']:.4f}")
    print(f"Recall: {metrics_dict['recall']:.4f}")
    print("================================\n")

    print("\n===== IMPACTO FINANCEIRO =====")
    print(f"Lucro estimado: {lucro}")
    print(f"Prejuízo estimado: {prejuizo}")
    print(f"Resultado líquido: {resultado}")
    print("================================\n")

    # =============================
    # MLflow
    # =============================
    with mlflow.start_run() as run:

        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("max_iter", 3000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("threshold", float(best_threshold))

        mlflow.log_metric("auc", float(metrics_dict["auc"]))
        mlflow.log_metric("precision", float(metrics_dict["precision"]))
        mlflow.log_metric("recall", float(metrics_dict["recall"]))
        mlflow.log_metric("positive_rate", float(y_pred.mean()))
        mlflow.log_metric("lucro", float(lucro))
        mlflow.log_metric("prejuizo", float(prejuizo))
        mlflow.log_metric("resultado", float(resultado))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")

        logger.info(
            "Treinamento concluído",
            extra={
                "run_id": run.info.run_id,
                "metrics": metrics_dict,
                "resultado": resultado,
            },
        )


if __name__ == "__main__":
    train()