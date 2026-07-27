from contextlib import asynccontextmanager
from typing import Dict

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from src.api.config import api_settings
from src.api.schemas import PredictRequest, PredictResponse
from src.api.services import ModelService
from src.logger import get_logger, log_inference, set_correlation_id
from src.monitoring.metrics import ModelMetrics
from src.monitoring.prometheus import PrometheusMetrics

load_dotenv()

logger = get_logger(__name__)

model_service = ModelService()
metrics = ModelMetrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.load_model()
    except FileNotFoundError:
        logger.warning("Nenhum modelo encontrado durante inicialização.")

    yield


app = FastAPI(
    title=api_settings.title,
    version=api_settings.version,
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def readiness_check() -> Dict[str, str]:
    if model_service.model is None:
        return {"status": "not_ready"}
    return {"status": "ready"}


@app.get("/version")
def version() -> Dict[str, str]:
    return {
        "version": api_settings.version,
        "service": api_settings.service_name,
    }


@app.get("/metrics")
def metrics_endpoint() -> str:
    return PrometheusMetrics(metrics).render()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, request_obj: Request) -> PredictResponse:
    correlation_id = request_obj.headers.get(
        "x-request-id") or set_correlation_id()
    request_obj.state.correlation_id = correlation_id

    try:
        if model_service.model is None:
            model_service.load_model()

        features = pd.DataFrame([request.model_dump()])
        probability = float(
            model_service.model.predict_proba(features)[:, 1][0])
        prediction = int(model_service.model.predict(features)[0])

        log_inference(
            request_id=correlation_id,
            model="champion",
            prediction=probability,
            latency_ms=0.0,
            features=request.model_dump(),
            threshold=0.5,
        )
        metrics.record_inference(
            probability=probability, prediction=prediction)

        return PredictResponse(
            prediction=prediction,
            probability=probability,
        )

    except Exception as error:
        logger.error(
            "Falha na inferência",
            extra={"error": str(error), "request_id": correlation_id},
        )

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )
