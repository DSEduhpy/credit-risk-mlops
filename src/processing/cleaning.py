from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET_COLUMN
from src.logger import get_logger

logger = get_logger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a limpeza do dataset bruto.

    Etapas:
        1. Criação da variável alvo (Target Engineering)
        2. Remoção de colunas irrelevantes
        3. Remoção de duplicatas
        4. Tratamento de valores ausentes
        5. Criação de features auxiliares
    """

    logger.info("Iniciando limpeza de dados")

    # ==========================================================
    # Evita modificar o DataFrame original recebido pela função
    # ==========================================================
    df = df.copy()

    initial_shape = df.shape

    # ==========================================================
    # 1. TARGET ENGINEERING
    # ==========================================================
    # Converte o status do empréstimo em uma variável binária:
    #
    # Fully Paid  -> 0 (Bom pagador)
    # Charged Off -> 1 (Inadimplente)
    #
    # Após criar a variável alvo, a coluna original é removida
    # para evitar Data Leakage.
    # ==========================================================
    if "loan_status" in df.columns:

        logger.info("Criando variável alvo")

        target_mapping = {
            "Fully Paid": 0,
            "Charged Off": 1,
        }

        # Mapeia os valores textuais para 0 e 1
        df[TARGET_COLUMN] = df["loan_status"].map(target_mapping)

        # Remove registros cujo status não foi mapeado
        df = df.dropna(subset=[TARGET_COLUMN])

        # Converte definitivamente para inteiro
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

        # Remove a coluna original
        df = df.drop(columns=["loan_status"])

        logger.info(
            "Target criada",
            extra={
                "target": TARGET_COLUMN,
                "positives": int(df[TARGET_COLUMN].sum()),
                "rows": len(df),
            },
        )

    # ==========================================================
    # 2. REMOÇÃO DE COLUNAS IRRELEVANTES
    # ==========================================================
    if "SK_ID_CURR" in df.columns:
        df = df.drop(columns=["SK_ID_CURR"])

    # ==========================================================
    # 3. REMOÇÃO DE DUPLICATAS
    # ==========================================================
    df = df.drop_duplicates()

    # ==========================================================
    # 4. REMOÇÃO DE COLUNAS COM MUITOS NULOS
    # Remove colunas com mais de 90% de valores ausentes.
    # ==========================================================
    missing_rate = df.isna().mean()

    high_missing_cols = missing_rate[missing_rate > 0.90].index.tolist()

    if high_missing_cols:

        df = df.drop(columns=high_missing_cols)

        logger.info(
            "Removendo colunas com alto percentual de missing",
            extra={"columns": high_missing_cols},
        )

    # ==========================================================
    # 5. IMPUTAÇÃO DE VALORES NUMÉRICOS
    #
    # A mediana é utilizada por ser robusta contra outliers.
    # A coluna alvo NÃO deve sofrer imputação.
    # ==========================================================
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:

        if col == TARGET_COLUMN:
            continue

        median_value = df[col].median()

        if pd.isna(median_value):
            median_value = 0

        df[col] = df[col].fillna(median_value)

    # ==========================================================
    # 6. IMPUTAÇÃO DE VARIÁVEIS CATEGÓRICAS
    #
    # Valores ausentes recebem a categoria "missing".
    # ==========================================================
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        df[col] = df[col].fillna("missing")

    # ==========================================================
    # 7. FEATURE AUXILIAR
    #
    # Aplica transformação logarítmica para reduzir assimetria
    # da distribuição da renda.
    # ==========================================================
    if "AMT_INCOME_TOTAL" in df.columns:
        df["AMT_INCOME_TOTAL_LOG"] = np.log1p(df["AMT_INCOME_TOTAL"])

    final_shape = df.shape

    logger.info(
        "Limpeza concluída",
        extra={
            "initial_shape": initial_shape,
            "final_shape": final_shape,
            "rows_removed": initial_shape[0] - final_shape[0],
            "columns_removed": initial_shape[1] - final_shape[1],
        },
    )

    return df
