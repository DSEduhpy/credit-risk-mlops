"""
Validação de qualidade de dados.

Implementa verificações de completude, consistência e qualidade geral.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

# Thresholds de qualidade
MISSING_CRITICAL_THRESHOLD = 0.5  # 50%
MISSING_WARNING_THRESHOLD = 0.1   # 10%
QUALITY_HEALTHY_THRESHOLD = 80
QUALITY_WARNING_THRESHOLD = 60


def validate_missing_data(
    df: pd.DataFrame,
    critical_columns: Optional[List[str]] = None,
    missing_threshold: float = MISSING_CRITICAL_THRESHOLD,
) -> Dict[str, Dict]:
    """
    Valida percentual de dados faltantes por coluna.

    Args:
        df: DataFrame a validar
        critical_columns: Colunas consideradas críticas
        missing_threshold: Threshold para bloqueio

    Returns:
        Dicionário com estatísticas de missing
    """
    if critical_columns is None:
        critical_columns = ["TARGET", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "AMT_CREDIT"]

    results = {}
    critical_issues = []

    for col in df.columns:
        total_count = len(df)
        missing_count = df[col].isnull().sum()
        missing_percentage = missing_count / total_count if total_count > 0 else 0

        is_critical = col in critical_columns
        is_above_threshold = missing_percentage > missing_threshold

        # Determinar severidade
        if is_critical and is_above_threshold:
            severity = "critical"
            critical_issues.append(col)
        elif missing_percentage > MISSING_WARNING_THRESHOLD:
            severity = "warning"
        else:
            severity = "ok"

        results[col] = {
            "total_count": total_count,
            "missing_count": int(missing_count),
            "missing_percentage": float(missing_percentage),
            "is_critical_column": is_critical,
            "severity": severity,
        }

    # Bloquear execução se houver problemas críticos
    if critical_issues:
        error_msg = f"Colunas críticas com missing excessivo: {critical_issues}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return results


def calculate_completeness_score(df: pd.DataFrame) -> float:
    """
    Calcula score de completude do dataset (0-100).

    Args:
        df: DataFrame a avaliar

    Returns:
        Score de completude
    """
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = (total_cells - missing_cells) / total_cells if total_cells > 0 else 1.0

    return float(completeness * 100)


def calculate_consistency_score(df: pd.DataFrame) -> float:
    """
    Calcula score de consistência dos dados (0-100).

    Args:
        df: DataFrame a avaliar

    Returns:
        Score de consistência
    """
    score = 100.0

    # Penalizar duplicatas
    duplicate_rate = df.duplicated().sum() / len(df) if len(df) > 0 else 0
    score -= duplicate_rate * 50  # Penalidade de até 50 pontos

    # Penalizar valores negativos em colunas que não deveriam ter
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if "AMT_" in col or "CNT_" in col:  # Colunas que devem ser positivas
            negative_rate = (df[col] < 0).sum() / len(df) if len(df) > 0 else 0
            score -= negative_rate * 20  # Penalidade de até 20 pontos por coluna

    return max(0.0, min(100.0, score))


def calculate_uniqueness_score(df: pd.DataFrame) -> float:
    """
    Calcula score de unicidade dos dados (0-100).

    Args:
        df: DataFrame a avaliar

    Returns:
        Score de unicidade
    """
    score = 100.0

    # Penalizar colunas com baixa variabilidade
    for col in df.select_dtypes(include=[np.number]).columns:
        unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 1.0
        if unique_ratio < 0.01:  # Menos de 1% únicos
            score -= 10  # Penalidade por coluna com baixa variabilidade

    return max(0.0, min(100.0, score))


def calculate_quality_score(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula score geral de qualidade de dados (0-100).

    Args:
        df: DataFrame a avaliar

    Returns:
        Dicionário com scores parciais e final
    """
    completeness = calculate_completeness_score(df)
    consistency = calculate_consistency_score(df)
    uniqueness = calculate_uniqueness_score(df)

    # Score final ponderado
    final_score = (completeness * 0.5) + (consistency * 0.3) + (uniqueness * 0.2)

    # Classificação
    if final_score >= QUALITY_HEALTHY_THRESHOLD:
        classification = "healthy"
    elif final_score >= QUALITY_WARNING_THRESHOLD:
        classification = "warning"
    else:
        classification = "critical"

    results = {
        "completeness_score": completeness,
        "consistency_score": consistency,
        "uniqueness_score": uniqueness,
        "final_score": final_score,
        "classification": classification,
    }

    logger.info(f"Score de qualidade calculado: {final_score:.1f} ({classification})")

    return results


def validate_data_quality(
    df: pd.DataFrame,
    critical_columns: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Executa validação completa de qualidade de dados.

    Args:
        df: DataFrame a validar
        critical_columns: Colunas críticas para validação de missing

    Returns:
        Dicionário consolidado com resultados
    """
    logger.info("Iniciando validação de qualidade de dados")

    results = {
        "missing_validation": validate_missing_data(df, critical_columns),
        "quality_score": calculate_quality_score(df),
    }

    quality_classification = results["quality_score"]["classification"]

    if quality_classification == "critical":
        logger.error("Qualidade de dados crítica detectada")
        raise ValueError("Qualidade de dados abaixo do threshold crítico")

    logger.info("Validação de qualidade concluída")
    return results