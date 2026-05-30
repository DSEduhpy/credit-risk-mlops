import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import (
    FEATURES_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def load_features() -> pd.DataFrame:
    """
    Carrega dataset de features processadas.
    """

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {FEATURES_PATH}"
        )

    return pd.read_parquet(FEATURES_PATH)


def split_data(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
):
    """
    Realiza split treino/teste.
    """

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Coluna alvo '{TARGET_COLUMN}' não encontrada"
        )

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN].astype(int)

    stratify_target = y if stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )