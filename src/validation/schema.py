"""
Validação de schema e estrutura de dados.

Implementa validações de schema, tipos de dados e estrutura esperada.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

# Schema esperado para dataset de risco de crédito
EXPECTED_SCHEMA = {
    "TARGET": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "AMT_INCOME_TOTAL": {"dtype": "float64", "nullable": False, "min": 0},
    "DAYS_BIRTH": {"dtype": "int64", "nullable": False, "max": 0},  # Valores negativos
    "DAYS_EMPLOYED": {"dtype": "int64", "nullable": True},
    "AMT_CREDIT": {"dtype": "float64", "nullable": False, "min": 0},
    "AMT_ANNUITY": {"dtype": "float64", "nullable": True, "min": 0},
    "AMT_GOODS_PRICE": {"dtype": "float64", "nullable": True, "min": 0},
    "REGION_POPULATION_RELATIVE": {"dtype": "float64", "nullable": False, "min": 0, "max": 1},
    "DAYS_ID_PUBLISH": {"dtype": "int64", "nullable": False},
    "FLAG_MOBIL": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_EMP_PHONE": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_WORK_PHONE": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_CONT_MOBILE": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_PHONE": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_EMAIL": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "OCCUPATION_TYPE": {"dtype": "object", "nullable": True},
    "CNT_FAM_MEMBERS": {"dtype": "float64", "nullable": False, "min": 1},
    "REGION_RATING_CLIENT": {"dtype": "int64", "nullable": False, "categories": [1, 2, 3]},
    "REGION_RATING_CLIENT_W_CITY": {"dtype": "int64", "nullable": False, "categories": [1, 2, 3]},
    "WEEKDAY_APPR_PROCESS_START": {"dtype": "object", "nullable": False},
    "HOUR_APPR_PROCESS_START": {"dtype": "int64", "nullable": False, "min": 0, "max": 23},
    "REG_REGION_NOT_LIVE_REGION": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "REG_REGION_NOT_WORK_REGION": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "LIVE_REGION_NOT_WORK_REGION": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "REG_CITY_NOT_LIVE_CITY": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "REG_CITY_NOT_WORK_CITY": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "LIVE_CITY_NOT_WORK_CITY": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "ORGANIZATION_TYPE": {"dtype": "object", "nullable": False},
    "EXT_SOURCE_1": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "EXT_SOURCE_2": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "EXT_SOURCE_3": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "APARTMENTS_AVG": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "BASEMENTAREA_AVG": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "YEARS_BEGINEXPLUATATION_AVG": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "YEARS_BUILD_AVG": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "COMMONAREA_AVG": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "ELEVATORS_AVG": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "ENTRANCES_AVG": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "FLOORSMAX_AVG": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "FLOORSMIN_AVG": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "LANDAREA_AVG": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "LIVINGAPARTMENTS_AVG": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "LIVINGAREA_AVG": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "NONLIVINGAPARTMENTS_AVG": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "NONLIVINGAREA_AVG": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "APARTMENTS_MODE": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "BASEMENTAREA_MODE": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "YEARS_BEGINEXPLUATATION_MODE": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "YEARS_BUILD_MODE": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "COMMONAREA_MODE": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "ELEVATORS_MODE": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "ENTRANCES_MODE": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "FLOORSMAX_MODE": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "FLOORSMIN_MODE": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "LANDAREA_MODE": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "LIVINGAPARTMENTS_MODE": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "LIVINGAREA_MODE": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "NONLIVINGAPARTMENTS_MODE": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "NONLIVINGAREA_MODE": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "APARTMENTS_MEDI": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "BASEMENTAREA_MEDI": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "YEARS_BEGINEXPLUATATION_MEDI": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "YEARS_BUILD_MEDI": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "COMMONAREA_MEDI": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "ELEVATORS_MEDI": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "ENTRANCES_MEDI": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "FLOORSMAX_MEDI": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "FLOORSMIN_MEDI": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "LANDAREA_MEDI": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "LIVINGAPARTMENTS_MEDI": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "LIVINGAREA_MEDI": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "NONLIVINGAPARTMENTS_MEDI": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "NONLIVINGAREA_MEDI": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "FONDKAPREMONT_MODE": {"dtype": "object", "nullable": True},
    "HOUSETYPE_MODE": {"dtype": "object", "nullable": True},
    "TOTALAREA_MODE": {"dtype": "float64", "nullable": True, "min": 0, "max": 1},
    "WALLSMATERIAL_MODE": {"dtype": "object", "nullable": True},
    "EMERGENCYSTATE_MODE": {"dtype": "object", "nullable": True},
    "OBS_30_CNT_SOCIAL_CIRCLE": {"dtype": "float64", "nullable": True, "min": 0},
    "DEF_30_CNT_SOCIAL_CIRCLE": {"dtype": "float64", "nullable": True, "min": 0},
    "OBS_60_CNT_SOCIAL_CIRCLE": {"dtype": "float64", "nullable": True, "min": 0},
    "DEF_60_CNT_SOCIAL_CIRCLE": {"dtype": "float64", "nullable": True, "min": 0},
    "DAYS_LAST_PHONE_CHANGE": {"dtype": "float64", "nullable": True},
    "FLAG_DOCUMENT_2": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_3": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_4": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_5": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_6": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_7": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_8": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_9": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_10": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_11": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_12": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_13": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_14": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_15": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_16": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_17": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_18": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_19": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_20": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "FLAG_DOCUMENT_21": {"dtype": "int64", "nullable": False, "categories": [0, 1]},
    "AMT_REQ_CREDIT_BUREAU_HOUR": {"dtype": "float64", "nullable": True, "min": 0},
    "AMT_REQ_CREDIT_BUREAU_DAY": {"dtype": "float64", "nullable": True, "min": 0},
    "AMT_REQ_CREDIT_BUREAU_WEEK": {"dtype": "float64", "nullable": True, "min": 0},
    "AMT_REQ_CREDIT_BUREAU_MON": {"dtype": "float64", "nullable": True, "min": 0},
    "AMT_REQ_CREDIT_BUREAU_QRT": {"dtype": "float64", "nullable": True, "min": 0},
    "AMT_REQ_CREDIT_BUREAU_YEAR": {"dtype": "float64", "nullable": True, "min": 0},
}

REQUIRED_COLUMNS = [
    "loan_amnt",
    "annual_inc",
    "default",
]


def validate_required_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Valida presença de colunas obrigatórias.

    Args:
        df: DataFrame a validar

    Returns:
        Tuple (is_valid, missing_columns)
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    is_valid = len(missing) == 0

    if not is_valid:
        logger.error(f"Colunas obrigatórias ausentes: {missing}")
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    return is_valid, missing


def validate_column_types(df):
    EXPECTED_DTYPES = {
    "loan_amnt": "float64",
    "annual_inc": "float64",
    "default": "int64",
    }

    for col, expected_dtype in EXPECTED_DTYPES.items():

        if col not in df.columns:
            continue

        actual_dtype = str(df[col].dtype)

        if actual_dtype != expected_dtype:
            raise TypeError(
                f"Coluna '{col}' deveria ser "
                f"{expected_dtype} mas recebeu "
                f"{actual_dtype}"
            )

    return True


def validate_ranges(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Valida ranges de valores numéricos.

    Args:
        df: DataFrame a validar

    Returns:
        Dicionário com resultados da validação
    """
    results = {}

    for col, expected in EXPECTED_SCHEMA.items():
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        col_min = expected.get("min")
        col_max = expected.get("max")

        if col_min is not None:
            violations_min = (df[col] < col_min).sum()
        else:
            violations_min = 0

        if col_max is not None:
            violations_max = (df[col] > col_max).sum()
        else:
            violations_max = 0

        total_violations = violations_min + violations_max
        is_valid = total_violations == 0

        results[col] = {
            "min_expected": col_min,
            "max_expected": col_max,
            "violations_min": int(violations_min),
            "violations_max": int(violations_max),
            "total_violations": int(total_violations),
            "is_valid": is_valid,
        }

        if not is_valid:
            logger.warning(f"Violação de range em {col}: {total_violations} valores fora do esperado")

    return results


def validate_categories(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Valida valores categóricos.

    Args:
        df: DataFrame a validar

    Returns:
        Dicionário com resultados da validação
    """
    results = {}

    for col, expected in EXPECTED_SCHEMA.items():
        if col not in df.columns or "categories" not in expected:
            continue

        expected_cats = set(expected["categories"])
        actual_unique = set(df[col].dropna().unique())

        invalid_values = actual_unique - expected_cats
        is_valid = len(invalid_values) == 0

        results[col] = {
            "expected_categories": list(expected_cats),
            "invalid_values": list(invalid_values),
            "is_valid": is_valid,
        }

        if not is_valid:
            logger.warning(f"Valores inválidos em {col}: {invalid_values}")

    return results


def validate_cardinality(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Valida cardinalidade de colunas categóricas.

    Args:
        df: DataFrame a validar

    Returns:
        Dicionário com resultados da validação
    """
    results = {}

    for col in df.select_dtypes(include=["object", "category"]).columns:
        unique_count = df[col].nunique()
        total_count = len(df)

        # Cardinalidade muito baixa pode indicar problema
        # Cardinalidade muito alta pode indicar dados únicos
        is_valid = 1 < unique_count < total_count * 0.9

        results[col] = {
            "unique_count": int(unique_count),
            "total_count": total_count,
            "cardinality_ratio": unique_count / total_count,
            "is_valid": is_valid,
        }

        if not is_valid:
            logger.warning(f"Cardinalidade suspeita em {col}: {unique_count} valores únicos")

    return results


def validate_schema(df):

    if df.empty:
        raise ValueError("DataFrame vazio")

    validate_required_columns(df)
    validate_column_types(df)

    results = {
        "required_columns": {
            "is_valid": True,
            "missing": [],
        },
        "column_types": {
            "is_valid": True,
        },
        "ranges": validate_ranges(df),
        "categories": validate_categories(df),
        "cardinality": validate_cardinality(df),
    }

    logger.info("Validação de schema concluída")

    return results


def detect_schema_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detecta mudanças de schema entre datasets.

    Args:
        reference_df: Dataset de referência
        current_df: Dataset atual

    Returns:
        Dicionário com mudanças detectadas
    """
    ref_cols = set(reference_df.columns)
    curr_cols = set(current_df.columns)

    added_columns = list(curr_cols - ref_cols)
    removed_columns = list(ref_cols - curr_cols)
    common_columns = ref_cols & curr_cols

    dtype_changes = []
    for col in common_columns:
        ref_dtype = str(reference_df[col].dtype)
        curr_dtype = str(current_df[col].dtype)
        if ref_dtype != curr_dtype:
            dtype_changes.append({
                "column": col,
                "reference_dtype": ref_dtype,
                "current_dtype": curr_dtype,
            })

    results = {
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "dtype_changes": dtype_changes,
    }

    # Log mudanças críticas
    if removed_columns:
        logger.warning(f"Colunas removidas: {removed_columns}")
    if dtype_changes:
        logger.warning(f"Mudanças de tipo: {dtype_changes}")

    return results