"""
Validações de expectativas e regras de negócio.

Implementa validações específicas do domínio de risco de crédito.
"""

from typing import Dict, List

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def validate_business_rules(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Valida regras de negócio específicas do domínio.

    Args:
        df: DataFrame a validar

    Returns:
        Dicionário com resultados das validações
    """
    results = {}

    # Regra 1: Renda deve ser positiva
    if "AMT_INCOME_TOTAL" in df.columns:
        invalid_income = (df["AMT_INCOME_TOTAL"] <= 0).sum()
        results["income_positive"] = {
            "description": "AMT_INCOME_TOTAL deve ser > 0",
            "violations": int(invalid_income),
            "is_valid": invalid_income == 0,
        }

    # Regra 2: Idade deve ser válida (DAYS_BIRTH negativo representa idade)
    if "DAYS_BIRTH" in df.columns:
        invalid_age = (df["DAYS_BIRTH"] >= 0).sum()  # Deve ser negativo
        results["age_valid"] = {
            "description": "DAYS_BIRTH deve ser < 0 (idade válida)",
            "violations": int(invalid_age),
            "is_valid": invalid_age == 0,
        }

    # Regra 3: TARGET deve ser binário
    if "TARGET" in df.columns:
        invalid_target = df[~df["TARGET"].isin([0, 1])]["TARGET"].count()
        results["target_binary"] = {
            "description": "TARGET deve ser 0 ou 1",
            "violations": int(invalid_target),
            "is_valid": invalid_target == 0,
        }

    # Regra 4: Valor do crédito deve ser positivo
    if "AMT_CREDIT" in df.columns:
        invalid_credit = (df["AMT_CREDIT"] <= 0).sum()
        results["credit_positive"] = {
            "description": "AMT_CREDIT deve ser > 0",
            "violations": int(invalid_credit),
            "is_valid": invalid_credit == 0,
        }

    # Regra 5: Número de familiares deve ser >= 1
    if "CNT_FAM_MEMBERS" in df.columns:
        invalid_family = (df["CNT_FAM_MEMBERS"] < 1).sum()
        results["family_members_valid"] = {
            "description": "CNT_FAM_MEMBERS deve ser >= 1",
            "violations": int(invalid_family),
            "is_valid": invalid_family == 0,
        }

    # Regra 6: Scores externos devem estar entre 0 e 1
    external_sources = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    for col in external_sources:
        if col in df.columns:
            invalid_scores = ((df[col] < 0) | (df[col] > 1)).sum()
            results[f"{col}_range"] = {
                "description": f"{col} deve estar entre 0 e 1",
                "violations": int(invalid_scores),
                "is_valid": invalid_scores == 0,
            }

    return results


def validate_data_consistency(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Valida consistência interna dos dados.

    Args:
        df: DataFrame a validar

    Returns:
        Dicionário com resultados das validações
    """
    results = {}

    # Consistência 1: Preço dos bens não deve exceder valor do crédito
    if "AMT_GOODS_PRICE" in df.columns and "AMT_CREDIT" in df.columns:
        inconsistent_goods = (df["AMT_GOODS_PRICE"] > df["AMT_CREDIT"]).sum()
        results["goods_price_consistency"] = {
            "description": "AMT_GOODS_PRICE não deve exceder AMT_CREDIT",
            "violations": int(inconsistent_goods),
            "is_valid": inconsistent_goods == 0,
        }

    # Consistência 2: Anuidade não deve exceder renda
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        inconsistent_annuity = (df["AMT_ANNUITY"] > df["AMT_INCOME_TOTAL"]).sum()
        results["annuity_income_consistency"] = {
            "description": "AMT_ANNUITY não deve exceder AMT_INCOME_TOTAL",
            "violations": int(inconsistent_annuity),
            "is_valid": inconsistent_annuity == 0,
        }

    # Consistência 3: Flags de documentos devem ser consistentes
    document_flags = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]
    available_flags = [col for col in document_flags if col in df.columns]

    if available_flags:
        # Pelo menos um documento deve ser fornecido (exceto FLAG_DOCUMENT_2 que é especial)
        has_any_document = df[available_flags].any(axis=1).sum()
        total_records = len(df)
        results["document_flags_consistency"] = {
            "description": "Pelo menos um documento deve ser marcado",
            "violations": int(total_records - has_any_document),
            "is_valid": has_any_document == total_records,
        }

    return results


def validate_expectations(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Executa validação completa de expectativas e regras de negócio.

    Args:
        df: DataFrame a validar

    Returns:
        Dicionário consolidado com resultados
    """
    logger.info("Iniciando validação de expectativas de negócio")

    results = {
        "business_rules": validate_business_rules(df),
        "data_consistency": validate_data_consistency(df),
    }

    # Verificar se há violações críticas
    critical_violations = []
    for category, validations in results.items():
        for rule_name, rule_result in validations.items():
            if not rule_result["is_valid"] and rule_result["violations"] > 0:
                critical_violations.append(f"{category}.{rule_name}")

    if critical_violations:
        logger.warning(f"Violações de expectativas detectadas: {critical_violations}")
        # Não bloquear execução, apenas alertar

    logger.info("Validação de expectativas concluída")
    return results