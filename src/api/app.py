from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import MLFLOW_TRACKING_URI, MODEL_PATH
from src.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

app = FastAPI(title="Credit Risk MLOps API")


def load_model() -> Any:
    if MODEL_PATH.exists():
        logger.info("Carregando modelo local", extra={"model_path": str(MODEL_PATH)})
        return joblib.load(MODEL_PATH)

    logger.info("Tentando carregar modelo do MLflow", extra={"tracking_uri": MLFLOW_TRACKING_URI})
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_candidates = list(Path(MLFLOW_TRACKING_URI).rglob("*/artifacts/model"))
    if not model_candidates:
        raise FileNotFoundError("Modelo não encontrado nem em models/ nem no MLflow local.")

    model_path = sorted(model_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    logger.info("Carregando modelo do MLflow", extra={"mlflow_model_path": str(model_path)})
    return mlflow.sklearn.load_model(str(model_path))


model = load_model()


class PredictRequest(BaseModel):
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        features = pd.DataFrame([request.features])
        probability = float(model.predict_proba(features)[:, 1][0])
        prediction = int(model.predict(features)[0])
        return PredictResponse(prediction=prediction, probability=probability)
    except Exception as error:
        logger.error("Falha na inferência", extra={"error": str(error)})
        raise HTTPException(status_code=400, detail="Erro ao processar a requisição de inferência.")
