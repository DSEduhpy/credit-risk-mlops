"""Schemas Pydantic para as rotas da API de inferência."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Payload esperado para uma solicitação de predição."""

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
    """Resposta retornada após a inferência."""

    prediction: int
    probability: float
