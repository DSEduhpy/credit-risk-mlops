"""
Módulo de ingestão de dados.
"""

import pandas as pd
import numpy as np

from src.logger import get_logger
from src.config import RAW_CSV_PATH, RAW_DATA_PATH

logger = get_logger(__name__)


def generate_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Gera dataset sintético para CI/CD.
    """

    rng = np.random.default_rng(42)

    data = pd.DataFrame({
        "AMT_INCOME_TOTAL": rng.normal(150000, 50000, n_samples),
        "DAYS_BIRTH": rng.integers(-25000, -7000, n_samples),
        "TARGET": rng.integers(0, 2, n_samples),
        "CODE_GENDER": rng.choice(["M", "F"], n_samples),
        "NAME_CONTRACT_TYPE": rng.choice(
            ["Cash loans", "Revolving loans"],
            n_samples
        ),
    })

    return data


def load_data() -> None:

    logger.info("Iniciando ingestão de dados")

    # ==========================================================
    # Dataset real
    # ==========================================================
    if RAW_CSV_PATH.exists():

        logger.info(
            f"Lendo dataset real: {RAW_CSV_PATH}"
        )

        df = pd.read_csv(RAW_CSV_PATH)

    # ==========================================================
    # Dataset sintético para CI
    # ==========================================================
    else:

        logger.warning(
            "Dataset real não encontrado. "
            "Gerando dataset sintético para CI/CD."
        )

        df = generate_synthetic_data()

    # ==========================================================
    # Salvamento
    # ==========================================================
    RAW_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        RAW_DATA_PATH,
        index=False,
    )

    logger.info("Ingestão concluída")


if __name__ == "__main__":
    load_data()