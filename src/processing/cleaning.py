from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET_COLUMN
from src.logger import get_logger

logger = get_logger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw loan dataset and return a cleaned DataFrame.
    """

    logger.info("Iniciando limpeza de dados")

    # evita mutação do dataframe original
    df = df.copy()

    initial_shape = df.shape

    # remover IDs irrelevantes
    if "SK_ID_CURR" in df.columns:
        df = df.drop(columns=["SK_ID_CURR"])

    # remover duplicatas
    df = df.drop_duplicates()

    # remover colunas com missing extremo
    missing_rate = df.isna().mean()
    high_missing_cols = missing_rate[missing_rate > 0.9].index.tolist()

    if high_missing_cols:
        df = df.drop(columns=high_missing_cols)

        logger.info(
            "Removendo colunas com alto missing",
            extra={"columns": high_missing_cols},
        )

    # preencher numéricos
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        if col == TARGET_COLUMN:
            continue

        median_value = df[col].median()

        # coluna completamente vazia
        if pd.isna(median_value):
            median_value = 0

        df[col] = df[col].fillna(median_value)

    # preencher categóricos
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        df[col] = df[col].fillna("missing")

    # feature opcional
    if "AMT_INCOME_TOTAL" in df.columns:
        df["AMT_INCOME_TOTAL_LOG"] = np.log1p(df["AMT_INCOME_TOTAL"])

    final_shape = df.shape

    logger.info(
        "Limpeza concluída",
        extra={
            "initial_shape": initial_shape,
            "final_shape": final_shape,
        },
    )

    return df