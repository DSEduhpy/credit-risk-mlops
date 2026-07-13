from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MLFLOW_TRACKING_URI, MODEL_PATH
from src.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

model = None


def load_model() -> Any:
    """
    Carrega modelo local ou do MLflow.
    """

    model_path = Path(MODEL_PATH)

    try:
        logger.info(
            "Tentando carregar modelo local",
            extra={"model_path": str(model_path)},
        )

        return joblib.load(model_path)

    except Exception:
        logger.warning("Modelo local não encontrado. Tentando MLflow.")

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        model_candidates = list(Path(MLFLOW_TRACKING_URI).rglob("*/artifacts/model"))

        if model_candidates:
            latest_model = sorted(
                model_candidates,
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[0]

            logger.info(
                "Carregando modelo do MLflow",
                extra={"mlflow_model_path": str(latest_model)},
            )

            return mlflow.sklearn.load_model(str(latest_model))

    except Exception:
        logger.warning("Falha ao carregar modelo do MLflow.")

    raise FileNotFoundError("Modelo não encontrado nem em models/ nem no MLflow local.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    try:
        model = load_model()
    except FileNotFoundError:
        logger.warning("Nenhum modelo encontrado durante inicialização.")
        model = None

    yield


app = FastAPI(
    title="Credit Risk MLOps API",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    loan_amnt: float = Field(gt=0)
    int_rate: float
    installment: float
    annual_inc: float = Field(gt=0)
    dti: float

    delinq_2yrs: float
    fico_range_low: float

    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float

    total_acc: float
    mort_acc: float
    pub_rec_bankruptcies: float

    home_ownership_encoded: float
    purpose_encoded: float

    loan_amnt_to_income: float
    fico_avg: float


class PredictResponse(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> Dict[str, str]:
    return {
        "version": "1.0.0",
        "service": "credit-risk-mlops-api",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    global model

    try:
        if model is None:
            model = load_model()

        features = pd.DataFrame([request.model_dump()])

        probability = float(model.predict_proba(features)[:, 1][0])

        prediction = int(model.predict(features)[0])

        return PredictResponse(
            prediction=prediction,
            probability=probability,
        )

    except Exception as error:
        import traceback

        traceback.print_exc()

        logger.error(
            "Falha na inferência",
            extra={"error": str(error)},
        )

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )
