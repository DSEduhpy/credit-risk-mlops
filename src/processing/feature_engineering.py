import pandas as pd
from src.config import CLEAN_DATA_PATH, FEATURES_PATH, TARGET_COLUMN
from src.logger import get_logger

logger = get_logger(__name__)


def build_features() -> None:
    logger.info("Iniciando engenharia de features")
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f"Dados limpos não encontrados em: {CLEAN_DATA_PATH}")

    df = pd.read_parquet(CLEAN_DATA_PATH)
    target = df[TARGET_COLUMN]
    df = df.drop(columns=[TARGET_COLUMN])

    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_columns:
        df = pd.get_dummies(df, columns=categorical_columns, dummy_na=True, drop_first=True)

    df[TARGET_COLUMN] = target
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PATH, index=False)

    logger.info(
        "Engenharia de features concluída",
        extra={
            "features_count": len(df.columns) - 1,
            "rows": len(df),
            "output_path": str(FEATURES_PATH),
        },
    )


if __name__ == "__main__":
    build_features()
