from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CLEAN_DATA_PATH, RAW_DATA_PATH, TARGET_COLUMN
from src.logger import get_logger

logger = get_logger(__name__)


def clean_data() -> None:
    logger.info("Iniciando limpeza de dados")
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dados brutos não encontrados em: {RAW_DATA_PATH}")

    df = pd.read_parquet(RAW_DATA_PATH)
    initial_shape = df.shape

    if "SK_ID_CURR" in df.columns:
        df = df.drop(columns=["SK_ID_CURR"])

    df = df.drop_duplicates()

    missing_rate = df.isna().mean()
    high_missing_cols = missing_rate[missing_rate > 0.9].index.tolist()
    if high_missing_cols:
        df = df.drop(columns=high_missing_cols)
        logger.info("Removendo colunas com alto missing", extra={"columns": high_missing_cols})

    for col in df.select_dtypes(include=["number"]).columns:
        if col == TARGET_COLUMN:
            continue
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("missing")

    if "AMT_INCOME_TOTAL" in df.columns:
        df["AMT_INCOME_TOTAL_LOG"] = np.log1p(df["AMT_INCOME_TOTAL"])

    df = df.dropna()
    final_shape = df.shape

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_DATA_PATH, index=False)

    logger.info(
        "Limpeza concluída",
        extra={
            "initial_shape": initial_shape,
            "final_shape": final_shape,
            "output_path": str(CLEAN_DATA_PATH),
        },
    )


if __name__ == "__main__":
    clean_data()
