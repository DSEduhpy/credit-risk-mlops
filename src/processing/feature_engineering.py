import pandas as pd
from src.config import CLEAN_DATA_PATH, FEATURES_PATH, TARGET_COLUMN
from src.logger import get_logger

logger = get_logger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função pura que aplica as transformações de engenharia de features.
    Não realiza operações de I/O.
    """
    # Cria uma cópia para evitar efeitos colaterais (side effects) no DF original
    df = df.copy()

    # ---- [Feature] Proporção do empréstimo em relação à renda ----
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["loan_amnt_to_income"] = (df["loan_amnt"] / df["annual_inc"]).fillna(0)

    # ---- [Feature] Média do Score FICO ----
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    # ---- [Feature] Encoding de Home Ownership ----
    if "home_ownership" in df.columns:
        mapping = {"MORTGAGE": 3, "OWN": 2, "RENT": 1}
        df["home_ownership_encoded"] = df["home_ownership"].map(mapping).fillna(0).astype(int)

    # ---- [Nova Feature] Encoding de Purpose ----
    if "purpose" in df.columns:
        # Converte para categoria e extrai os códigos numéricos ordinais
        df["purpose_encoded"] = df["purpose"].astype("category").cat.codes
    # ------------------------------------------------------------------

    if TARGET_COLUMN in df.columns:
        target = df[TARGET_COLUMN]
        df = df.drop(columns=[TARGET_COLUMN])
    else:
        target = None

    # Processamento de variáveis categóricas
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_columns:
        df = pd.get_dummies(df, columns=categorical_columns, dummy_na=True, drop_first=True)
        # Garante compatibilidade de nomes de colunas com o LightGBM
        df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)

    # Devolve o target ao DataFrame se ele existia originalmente
    if target is not None:
        df[TARGET_COLUMN] = target

    return df


def build_features() -> None:
    """
    Orquestrador do pipeline de features. Cuida de I/O e logging.
    """
    logger.info("Iniciando engenharia de features")
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f"Dados limpos não encontrados em: {CLEAN_DATA_PATH}")

    # I/O: Leitura
    df = pd.read_parquet(CLEAN_DATA_PATH)
    
    # Processamento através da função pura
    df_transformed = engineer_features(df)
    
    # I/O: Escrita
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_transformed.to_parquet(FEATURES_PATH, index=False)

    logger.info(
        "Engenharia de features concluída",
        extra={
            "features_count": len(df_transformed.columns) - 1,
            "rows": len(df_transformed),
            "output_path": str(FEATURES_PATH),
        },
    )


if __name__ == "__main__":
    build_features()