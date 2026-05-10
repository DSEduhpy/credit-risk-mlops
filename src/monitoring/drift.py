"""
Módulo de monitoramento de drift para sistemas de risco de crédito.

Implementa Population Stability Index (PSI) e Kolmogorov-Smirnov Test
para detecção de drift estatístico entre datasets de referência e produção.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import FEATURES_PATH, PROJECT_ROOT
from src.logger import get_logger

logger = get_logger(__name__)

# Paths para relatórios
REPORTS_PATH = PROJECT_ROOT.parent / "reports" / "drift"
FIGURES_PATH = REPORTS_PATH / "figures"

# Tentativa de importar bibliotecas opcionais
try:
    from scipy.stats import ks_2samp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy não disponível. KS Test será desabilitado.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib não disponível. Visualizações serão desabilitadas.")


def calculate_psi(
    reference: Union[pd.Series, np.ndarray],
    current: Union[pd.Series, np.ndarray],
    bins: int = 10,
) -> Tuple[float, str]:
    """
    Calcula Population Stability Index (PSI) entre duas distribuições.

    Args:
        reference: Dados de referência (treino/produção histórica)
        current: Dados atuais (produção nova)
        bins: Número de bins para discretização

    Returns:
        Tuple com (psi_value, drift_level)
    """
    # Converter para numpy arrays
    ref_data = np.asarray(reference)
    curr_data = np.asarray(current)

    # Remover NaNs
    ref_data = ref_data[~np.isnan(ref_data)]
    curr_data = curr_data[~np.isnan(curr_data)]

    if len(ref_data) == 0 or len(curr_data) == 0:
        return 0.0, "insufficient_data"

    # Criar bins automáticos
    all_data = np.concatenate([ref_data, curr_data])
    bin_edges = np.histogram_bin_edges(all_data, bins=bins)

    # Calcular percentuais
    ref_hist, _ = np.histogram(ref_data, bins=bin_edges)
    curr_hist, _ = np.histogram(curr_data, bins=bin_edges)

    # Evitar divisão por zero
    ref_perc = ref_hist / len(ref_data)
    curr_perc = curr_hist / len(curr_data)

    # Evitar zeros nos percentuais para log
    ref_perc = np.where(ref_perc == 0, 1e-10, ref_perc)
    curr_perc = np.where(curr_perc == 0, 1e-10, curr_perc)

    # Calcular PSI
    psi_components = (curr_perc - ref_perc) * np.log(curr_perc / ref_perc)
    psi_value = np.sum(psi_components)

    # Classificar nível de drift
    drift_level = classify_drift_level(psi_value)

    return float(psi_value), drift_level


def calculate_feature_psi(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature: str,
    bins: int = 10,
) -> Tuple[float, str]:
    """
    Calcula PSI para uma feature específica.

    Args:
        reference_df: DataFrame de referência
        current_df: DataFrame atual
        feature: Nome da feature
        bins: Número de bins

    Returns:
        Tuple com (psi_value, drift_level)
    """
    if feature not in reference_df.columns or feature not in current_df.columns:
        logger.warning(f"Feature {feature} não encontrada em ambos datasets")
        return 0.0, "feature_not_found"

    ref_series = reference_df[feature]
    curr_series = current_df[feature]

    return calculate_psi(ref_series, curr_series, bins)


def calculate_ks_test(
    reference: Union[pd.Series, np.ndarray],
    current: Union[pd.Series, np.ndarray],
) -> Tuple[float, float]:
    """
    Calcula Kolmogorov-Smirnov Test entre duas distribuições.

    Args:
        reference: Dados de referência
        current: Dados atuais

    Returns:
        Tuple com (ks_statistic, p_value)
    """
    if not SCIPY_AVAILABLE:
        logger.warning("KS Test não disponível (scipy não instalado)")
        return 0.0, 1.0

    ref_data = np.asarray(reference)
    curr_data = np.asarray(current)

    # Remover NaNs
    ref_data = ref_data[~np.isnan(ref_data)]
    curr_data = curr_data[~np.isnan(curr_data)]

    if len(ref_data) == 0 or len(curr_data) == 0:
        return 0.0, 1.0

    ks_stat, p_value = ks_2samp(ref_data, curr_data)
    return float(ks_stat), float(p_value)


def classify_drift_level(psi_value: float) -> str:
    """
    Classifica o nível de drift baseado no PSI.

    Args:
        psi_value: Valor do PSI

    Returns:
        Nível de drift: 'stable', 'moderate_drift', 'severe_drift'
    """
    if psi_value < 0.1:
        return "stable"
    elif psi_value <= 0.25:
        return "moderate_drift"
    else:
        return "severe_drift"


def plot_drift_distribution(
    reference: Union[pd.Series, np.ndarray],
    current: Union[pd.Series, np.ndarray],
    feature: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Gera visualização comparativa de distribuições.

    Args:
        reference: Dados de referência
        current: Dados atuais
        feature: Nome da feature
        save_path: Caminho para salvar figura (opcional)
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Visualização não disponível (matplotlib não instalado)")
        return

    ref_data = np.asarray(reference)
    curr_data = np.asarray(current)

    # Remover NaNs
    ref_data = ref_data[~np.isnan(ref_data)]
    curr_data = curr_data[~np.isnan(curr_data)]

    plt.figure(figsize=(10, 6))

    # Histogramas
    plt.hist(ref_data, alpha=0.7, label='Referência', bins=30, density=True)
    plt.hist(curr_data, alpha=0.7, label='Atual', bins=30, density=True)

    plt.title(f'Distribuição - {feature}')
    plt.xlabel('Valor')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualização salva em {save_path}")
    else:
        plt.show()

    plt.close()


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Gera relatório completo de drift para múltiplas features.

    Args:
        reference_df: DataFrame de referência
        current_df: DataFrame atual
        features: Lista de features para analisar (opcional)

    Returns:
        DataFrame com relatório de drift
    """
    if features is None:
        # Analisar apenas features numéricas presentes em ambos
        features = list(
            reference_df.select_dtypes(include=[np.number]).columns.intersection(
                current_df.select_dtypes(include=[np.number]).columns
            )
        )

    logger.info(f"Iniciando análise de drift para {len(features)} features")

    report_data = []

    for feature in features:
        logger.info(f"Analisando drift para feature: {feature}")

        # Calcular PSI
        psi_value, drift_level = calculate_feature_psi(
            reference_df, current_df, feature
        )

        # Calcular KS Test
        ks_stat, p_value = calculate_ks_test(
            reference_df[feature], current_df[feature]
        )

        # Verificar drift severo
        if drift_level == "severe_drift":
            logger.warning(f"Drift severo detectado em {feature} (PSI: {psi_value:.4f})")

        report_data.append({
            "feature": feature,
            "psi": psi_value,
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "drift_level": drift_level,
        })

        # Gerar visualização
        fig_path = FIGURES_PATH / f"{feature}_drift.png"
        plot_drift_distribution(
            reference_df[feature],
            current_df[feature],
            feature,
            save_path=fig_path,
        )

    return pd.DataFrame(report_data)


def save_drift_report(report_df: pd.DataFrame) -> None:
    """
    Salva relatório de drift em CSV e JSON.

    Args:
        report_df: DataFrame com relatório
    """
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    # Salvar CSV
    csv_path = REPORTS_PATH / "drift_report.csv"
    report_df.to_csv(csv_path, index=False)
    logger.info(f"Relatório CSV salvo em {csv_path}")

    # Salvar JSON
    json_path = REPORTS_PATH / "drift_report.json"
    report_dict = report_df.to_dict('records')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Relatório JSON salvo em {json_path}")


def run_drift_monitoring(
    reference_path: Path,
    current_path: Path,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Executa monitoramento completo de drift.

    Args:
        reference_path: Caminho para dataset de referência
        current_path: Caminho para dataset atual
        features: Features específicas para analisar

    Returns:
        DataFrame com relatório de drift
    """
    logger.info("Iniciando monitoramento de drift")

    # Carregar datasets
    reference_df = pd.read_parquet(reference_path)
    current_df = pd.read_parquet(current_path)

    logger.info(f"Dataset referência: {len(reference_df)} linhas")
    logger.info(f"Dataset atual: {len(current_df)} linhas")

    # Gerar relatório
    report_df = generate_drift_report(reference_df, current_df, features)

    # Salvar relatório
    save_drift_report(report_df)

    logger.info("Monitoramento de drift concluído")

    return report_df


def main() -> None:
    """
    Função principal para execução via linha de comando.
    """
    import sys

    if len(sys.argv) < 2:
        print("Uso: python src/monitoring/drift.py <current_data_path> [features...]")
        print("Exemplo: python src/monitoring/drift.py data/production/current.parquet")
        sys.exit(1)

    current_path = Path(sys.argv[1])
    features = sys.argv[2:] if len(sys.argv) > 2 else None

    # Usar FEATURES_PATH como referência
    reference_path = FEATURES_PATH

    try:
        report = run_drift_monitoring(reference_path, current_path, features)
        print("\n===== RELATÓRIO DE DRIFT =====")
        print(report.to_string(index=False))
    except Exception as e:
        logger.error(f"Erro no monitoramento de drift: {e}")
        raise


if __name__ == "__main__":
    main()
