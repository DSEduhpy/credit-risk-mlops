from pathlib import Path

import pandas as pd

from src.config import RAW_CSV_PATH, RAW_DATA_PATH
from src.logger import get_logger

logger = get_logger(__name__)


def load_data() -> None:
    logger.info("Iniciando ingestão de dados")
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado em: {RAW_CSV_PATH}")

    df = pd.read_csv(RAW_CSV_PATH)
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_DATA_PATH, index=False)

    logger.info(
        "Ingestão concluída",
        extra={
            "rows": len(df),
            "columns": len(df.columns),
            "output_path": str(RAW_DATA_PATH),
        },
    )


if __name__ == "__main__":
    load_data()
