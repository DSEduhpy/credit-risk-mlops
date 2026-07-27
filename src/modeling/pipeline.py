"""
Pipeline refatorado de treinamento e benchmark para risco de crédito.

A lógica de treinamento foi separada em componentes com uma única
responsabilidade: treinar modelos, otimizar thresholds, avaliar métricas
financeiras, persistir artefatos e registrar dados no MLflow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import mlflow
import pandas as pd

from src.config import MLFLOW_TRACKING_URI, settings
from src.evaluation.business_metrics import simulate_business_metrics
from src.evaluation.metrics import compute_metrics
from src.evaluation.threshold import optimize_threshold
from src.logger import get_logger
from src.modeling.data import load_features, split_data
from src.modeling.models.catboost import build_catboost_model
from src.modeling.models.lightgbm import build_lightgbm_model
from src.modeling.models.logistic import build_logistic_model
from src.modeling.models.xgboost import build_xgboost_model
from src.modeling.registry import ModelRegistry

logger = get_logger(__name__)

MODELS: dict[str, Callable[[], Any]] = {
    "logistic": build_logistic_model,
    "xgboost": build_xgboost_model,
    "lightgbm": build_lightgbm_model,
    "catboost": build_catboost_model,
}


@dataclass
class DatasetSplit:
    """Estrutura contendo divisão de treino e teste."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class ThresholdOptimizer:
    """Encapsula a otimização de threshold para decisão de aprovação."""

    def optimize(self, y_true: pd.Series, y_proba: pd.Series) -> tuple[float, Any]:
        """Retorna o melhor threshold e o resultado associado."""
        return optimize_threshold(
            y_true=y_true,
            y_proba=y_proba,
            custo_inadimplente=10_000,
            lucro_cliente=1_000,
        )


class BusinessEvaluator:
    """Responsável por calcular métricas técnicas e financeiras."""

    def evaluate(
        self, y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series
    ) -> dict[str, Any]:
        """Combina métricas técnicas e financeiras para um modelo."""
        metrics_dict = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        )
        business_metrics = simulate_business_metrics(
            y_true=y_true,
            y_pred=y_pred,
            custo_inadimplente=10_000,
            lucro_cliente=1_000,
        )
        return {
            "metrics": metrics_dict,
            "business_metrics": business_metrics,
        }


class Trainer:
    """Responsável apenas pelo treinamento de um modelo."""

    def __init__(
        self,
        threshold_optimizer: ThresholdOptimizer | None = None,
        business_evaluator: BusinessEvaluator | None = None,
    ) -> None:
        self.threshold_optimizer = threshold_optimizer or ThresholdOptimizer()
        self.business_evaluator = business_evaluator or BusinessEvaluator()

    def train_single_model(
        self, model_name: str, model_func: Callable[[], Any], data: DatasetSplit
    ) -> dict[str, Any]:
        """Treina um único modelo e retorna um resumo com métricas."""
        logger.info("Iniciando treinamento do modelo: %s", model_name)
        start_time = time.perf_counter()

        model = model_func()
        model.fit(data.x_train, data.y_train)

        y_proba = model.predict_proba(data.x_test)[:, 1]
        best_threshold, _ = self.threshold_optimizer.optimize(data.y_test, y_proba)
        y_pred = (y_proba >= best_threshold).astype(int)
        evaluation = self.business_evaluator.evaluate(data.y_test, y_pred, y_proba)

        training_time = time.perf_counter() - start_time
        logger.info(
            "Modelo %s finalizado",
            model_name,
            extra={
                "model": model_name,
                "training_time": training_time,
                "best_threshold": best_threshold,
                "resultado": evaluation["business_metrics"]["resultado"],
            },
        )

        return {
            "model_name": model_name,
            "model": model,
            "metrics": evaluation["metrics"],
            "business_metrics": evaluation["business_metrics"],
            "best_threshold": best_threshold,
            "training_time": training_time,
        }


class MLflowTracker:
    """Responsável apenas pelo registro de artefatos e métricas no MLflow."""

    def __init__(self, storage: "ModelStorage | None" = None) -> None:
        self.storage = storage or ModelStorage()

    def log_experiment(self, model_name: str, result: dict[str, Any]) -> None:
        """Registra o resultado de um experimento no MLflow."""
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            mlflow.log_param("threshold", float(result["best_threshold"]))
            mlflow.log_param("training_time", float(result["training_time"]))

            for key, value in result["metrics"].items():
                mlflow.log_metric(key, float(value))

            for key, value in result["business_metrics"].items():
                mlflow.log_metric(key, float(value))

            model_path = self.storage.persist_model(model_name, result["model"])
            mlflow.log_artifact(str(model_path), artifact_path="model")


class ModelStorage:
    """Responsável por persistir modelos treinados em disco."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or settings.paths.models)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def persist_model(self, model_name: str, model: Any) -> Path:
        """Salva o modelo em disco e retorna o caminho gerado."""
        model_path = self.base_dir / f"{model_name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        return model_path


class Leaderboard:
    """Responsável somente pela exibição do ranking final."""

    def render(self, results: list[dict[str, Any]]) -> None:
        """Imprime o leaderboard do benchmark."""
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
            resultado = result["business_metrics"].get("resultado", 0)
            print(
                f"{result['model_name']:<12} | "
                f"{auc:<6.4f} | "
                f"{precision:<10.4f} | "
                f"{recall:<8.4f} | "
                f"{resultado:<10.0f}"
            )

        print("=" * 60)


class PipelineRunner:
    """Orquestra o benchmark completo de modelos e a promoção do campeão."""

    def __init__(self) -> None:
        self.trainer = Trainer()
        self.tracker = MLflowTracker()
        self.storage = self.tracker.storage
        self.leaderboard = Leaderboard()
        self.registry = ModelRegistry("models/registry.json")

    def run(self) -> list[dict[str, Any]]:
        """Executa o pipeline completo e retorna os resultados dos modelos."""
        logger.info("Iniciando benchmark multi-modelo")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("credit_risk_benchmark")

        data = load_features()
        x_train, x_test, y_train, y_test = split_data(data)
        dataset = DatasetSplit(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )

        results: list[dict[str, Any]] = []
        for model_name, model_func in MODELS.items():
            result = self.trainer.train_single_model(model_name, model_func, dataset)
            self.tracker.log_experiment(model_name, result)
            model_path = self.storage.persist_model(model_name, result["model"])
            self.registry.register(
                model_name=model_name,
                model_path=model_path,
                stage="Development",
                metadata={
                    "business_result": result["business_metrics"].get("resultado", 0),
                    "auc": result["metrics"].get("auc", 0),
                },
            )
            results.append(result)

        self.leaderboard.render(results)

        if results:
            champion = max(
                results, key=lambda item: item["business_metrics"].get("resultado", 0)
            )
            self.registry.promote(champion["model_name"], target_stage="Production")
            logger.info(
                "Modelo campeão promovido para produção",
                extra={"model_name": champion["model_name"]},
            )

        logger.info("Benchmark concluído", extra={"num_models": len(results)})
        return results


def train_single_model(
    model_name: str, model_func: Callable[[], Any], data: DatasetSplit
) -> dict[str, Any]:
    """Compatibilidade com o módulo anterior: treina um modelo isolado."""
    return Trainer().train_single_model(model_name, model_func, data)


def log_to_mlflow(model_name: str, result: dict[str, Any]) -> None:
    """Compatibilidade com o módulo anterior: registra experimento no MLflow."""
    MLflowTracker().log_experiment(model_name, result)


def print_leaderboard(results: list[dict[str, Any]]) -> None:
    """Compatibilidade com o módulo anterior: exibe leaderboard."""
    Leaderboard().render(results)


def train() -> None:
    """Ponto de entrada compatível com o fluxo anterior."""
    PipelineRunner().run()
