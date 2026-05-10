"""
Orquestrador principal de validação de dados.

Coordena todas as validações e gera relatórios consolidados.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.config import PROJECT_ROOT
from src.logger import get_logger

from .schema import validate_schema, detect_schema_drift
from .quality import validate_data_quality
from .expectations import validate_expectations

logger = get_logger(__name__)

# Paths para relatórios
REPORTS_PATH = PROJECT_ROOT.parent / "reports" / "quality"


def generate_validation_report(
    schema_results: Dict,
    quality_results: Dict,
    expectations_results: Dict,
    schema_drift_results: Optional[Dict] = None,
) -> Dict:
    """
    Gera relatório consolidado de validação.

    Args:
        schema_results: Resultados da validação de schema
        quality_results: Resultados da validação de qualidade
        expectations_results: Resultados da validação de expectativas
        schema_drift_results: Resultados da detecção de drift de schema

    Returns:
        Dicionário com relatório completo
    """
    report = {
        "validation_summary": {
            "schema_validation": "passed" if all(
                all(v.get("is_valid", True) for v in cat.values())
                for cat in schema_results.values()
                if isinstance(cat, dict)
            ) else "failed",
            "quality_validation": quality_results["quality_score"]["classification"],
            "expectations_validation": "passed" if all(
                all(v.get("is_valid", True) for v in cat.values())
                for cat in expectations_results.values()
                if isinstance(cat, dict)
            ) else "warnings",
        },
        "details": {
            "schema": schema_results,
            "quality": quality_results,
            "expectations": expectations_results,
        },
    }

    if schema_drift_results:
        report["schema_drift"] = schema_drift_results

    return report


def save_validation_report(report: Dict) -> None:
    """
    Salva relatório de validação em JSON e CSV.

    Args:
        report: Relatório a salvar
    """
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    # Salvar JSON
    json_path = REPORTS_PATH / "quality_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Relatório JSON salvo em {json_path}")

    # Salvar CSV simplificado
    csv_path = REPORTS_PATH / "quality_report.csv"

    # Extrair métricas principais para CSV
    csv_data = []

    # Schema summary
    schema_status = report["validation_summary"]["schema_validation"]
    csv_data.append({
        "category": "schema",
        "metric": "overall_status",
        "value": schema_status,
        "details": ""
    })

    # Quality scores
    quality = report["details"]["quality"]["quality_score"]
    for key, value in quality.items():
        csv_data.append({
            "category": "quality",
            "metric": key,
            "value": value,
            "details": ""
        })

    # Expectations summary
    expectations_status = report["validation_summary"]["expectations_validation"]
    csv_data.append({
        "category": "expectations",
        "metric": "overall_status",
        "value": expectations_status,
        "details": ""
    })

    # Missing data summary
    missing = report["details"]["quality"]["missing_validation"]
    for col, stats in missing.items():
        if stats["severity"] != "ok":
            csv_data.append({
                "category": "missing",
                "metric": f"{col}_missing_pct",
                "value": stats["missing_percentage"],
                "details": stats["severity"]
            })

    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    logger.info(f"Relatório CSV salvo em {csv_path}")


def run_data_validation(
    df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    critical_columns: Optional[list] = None,
) -> Dict:
    """
    Executa pipeline completo de validação de dados.

    Args:
        df: DataFrame a validar
        reference_df: DataFrame de referência para comparação (opcional)
        critical_columns: Colunas consideradas críticas

    Returns:
        Relatório completo de validação
    """
    logger.info("Iniciando pipeline de validação de dados")

    # 1. Validação de schema
    schema_results = validate_schema(df)

    # 2. Validação de qualidade
    quality_results = validate_data_quality(df, critical_columns)

    # 3. Validação de expectativas
    expectations_results = validate_expectations(df)

    # 4. Detecção de schema drift (se referência fornecida)
    schema_drift_results = None
    if reference_df is not None:
        schema_drift_results = detect_schema_drift(reference_df, df)

    # 5. Gerar relatório consolidado
    report = generate_validation_report(
        schema_results,
        quality_results,
        expectations_results,
        schema_drift_results,
    )

    # 6. Salvar relatório
    save_validation_report(report)

    # 7. Log do resultado final
    overall_status = "PASSED"
    if report["validation_summary"]["schema_validation"] == "failed":
        overall_status = "FAILED - SCHEMA"
    elif report["validation_summary"]["quality_validation"] == "critical":
        overall_status = "FAILED - QUALITY"
    elif report["validation_summary"]["expectations_validation"] == "warnings":
        overall_status = "WARNINGS"

    logger.info(f"Validação concluída: {overall_status}")

    return report


def main() -> None:
    """
    Função principal para execução via linha de comando.
    """
    import sys

    if len(sys.argv) < 2:
        print("Uso: python src/validation/validator.py <data_path> [reference_path]")
        print("Exemplo: python src/validation/validator.py data/features/features.parquet")
        sys.exit(1)

    data_path = Path(sys.argv[1])
    reference_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    try:
        # Carregar dados
        df = pd.read_parquet(data_path)
        logger.info(f"Dataset carregado: {len(df)} linhas, {len(df.columns)} colunas")

        reference_df = None
        if reference_path and reference_path.exists():
            reference_df = pd.read_parquet(reference_path)
            logger.info(f"Dataset de referência carregado: {len(reference_df)} linhas")

        # Executar validação
        report = run_data_validation(df, reference_df)

        # Output no terminal
        print("\n===== RELATÓRIO DE VALIDAÇÃO =====")
        print(f"Schema: {report['validation_summary']['schema_validation'].upper()}")
        print(f"Quality: {report['validation_summary']['quality_validation'].upper()}")
        print(f"Expectations: {report['validation_summary']['expectations_validation'].upper()}")

        quality_score = report['details']['quality']['quality_score']
        print(f"\nScore de Qualidade: {quality_score['final_score']:.1f}/100 ({quality_score['classification']})")

        if "schema_drift" in report:
            drift = report["schema_drift"]
            if drift["added_columns"] or drift["removed_columns"] or drift["dtype_changes"]:
                print(f"\nSchema Drift Detectado:")
                if drift["added_columns"]:
                    print(f"  Colunas adicionadas: {drift['added_columns']}")
                if drift["removed_columns"]:
                    print(f"  Colunas removidas: {drift['removed_columns']}")
                if drift["dtype_changes"]:
                    print(f"  Mudanças de tipo: {len(drift['dtype_changes'])} colunas")

        print("\nRelatórios salvos em reports/quality/")

    except Exception as e:
        logger.error(f"Erro na validação: {e}")
        raise


if __name__ == "__main__":
    main()