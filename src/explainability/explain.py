"""
Módulo de Explainabilidade de Modelos com SHAP.

Este módulo implementa a explicabilidade do modelo de risco de crédito
utilizando SHAP (SHapley Additive exPlanations) para fornecer insights
sobre as decisões do modelo e a importância das features.

Functions:
    load_model: Carrega o modelo treinado do disco.
    load_data: Carrega o dataset de features.
    sample_data: Realiza amostragem dos dados para performance.
    create_explainer: Cria um objeto SHAP Explainer.
    generate_shap_values: Gera os valores SHAP para as amostras.
    plot_summary: Gera e salva o gráfico de resumo SHAP.
    plot_feature_importance: Gera e salva o gráfico de importância.
    main: Orquestra a execução completa do pipeline.
"""

import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional

from src.config import FEATURES_PATH, MODEL_PATH, TARGET_COLUMN, RANDOM_STATE
from src.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# CONSTANTES
# ============================================================================

MAX_SAMPLES = 2000
"""Número máximo de amostras para análise SHAP (performance safety)."""

FIGURE_DPI = 300
"""Resolução dos gráficos salvos em DPI."""

FIGURE_FORMAT = "png"
"""Formato dos gráficos salvos."""

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
"""Diretório base para salvar relatórios."""

FIGURES_DIR = REPORTS_DIR / "figures"
"""Diretório para salvar figuras."""


# ============================================================================
# FUNÇÕES
# ============================================================================


def load_model(model_path: Optional[Path] = None) -> object:
    """
    Carrega o modelo treinado do disco.

    Args:
        model_path: Caminho do arquivo do modelo. Se None, usa MODEL_PATH da config.

    Returns:
        O modelo carregado (Pipeline sklearn).

    Raises:
        FileNotFoundError: Se o arquivo do modelo não existir.

    Exemplo:
        >>> model = load_model()
        >>> type(model).__name__
        'Pipeline'
    """
    model_path = model_path or MODEL_PATH

    if not model_path.exists():
        logger.error(f"Arquivo de modelo não encontrado: {model_path}")
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")

    logger.info(f"Carregando modelo de: {model_path}")
    model = joblib.load(model_path)
    logger.info("Modelo carregado com sucesso")

    return model


def load_data(features_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Carrega o dataset de features do disco.

    Args:
        features_path: Caminho do arquivo de features. Se None, usa FEATURES_PATH da config.

    Returns:
        DataFrame com as features.

    Raises:
        FileNotFoundError: Se o arquivo de features não existir.
        ValueError: Se a coluna alvo não estiver presente.

    Exemplo:
        >>> df = load_data()
        >>> df.shape
        (10000, 25)
    """
    features_path = features_path or FEATURES_PATH

    if not features_path.exists():
        logger.error(f"Arquivo de features não encontrado: {features_path}")
        raise FileNotFoundError(f"Features não encontradas em: {features_path}")

    logger.info(f"Carregando features de: {features_path}")
    data = pd.read_parquet(features_path)

    if TARGET_COLUMN not in data.columns:
        logger.error(f"Coluna alvo '{TARGET_COLUMN}' não encontrada no dataset")
        raise ValueError(f"Coluna '{TARGET_COLUMN}' não encontrada")

    logger.info(f"Features carregadas: {data.shape[0]} linhas, {data.shape[1]} colunas")

    return data


def sample_data(
    data: pd.DataFrame,
    max_samples: int = MAX_SAMPLES,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Realiza amostragem dos dados para melhorar performance da análise SHAP.

    Se o dataset for maior que max_samples, realiza amostragem aleatória stratificada.
    Caso contrário, retorna o dataset completo.

    Args:
        data: DataFrame original.
        max_samples: Número máximo de amostras desejadas.
        random_state: Seed para reprodutibilidade.

    Returns:
        DataFrame amostrado ou original se menor que max_samples.

    Exemplo:
        >>> sampled = sample_data(df, max_samples=1000)
        >>> sampled.shape[0] <= 1000
        True
    """
    n_samples = len(data)

    if n_samples <= max_samples:
        logger.info(
            f"Dataset tem {n_samples} amostras (<=  {max_samples}). "
            "Usando dataset completo."
        )
        return data

    logger.info(
        f"Dataset tem {n_samples} amostras. "
        f"Amostrando {max_samples} para análise SHAP."
    )

    # Amostragem estratificada na coluna alvo para preservar distribuição
    # Usa groupby para garantir representação proporcional de cada classe
    if TARGET_COLUMN in data.columns:
        # Estratificado por classe
        stratified_samples = []
        for group in data[TARGET_COLUMN].unique():
            group_data = data[data[TARGET_COLUMN] == group]
            proportion = len(group_data) / len(data)
            group_sample_size = max(1, int(max_samples * proportion))
            group_samples = group_data.sample(
                n=min(group_sample_size, len(group_data)),
                random_state=random_state
            )
            stratified_samples.append(group_samples)
        sampled = pd.concat(stratified_samples).sample(
            frac=1, random_state=random_state  # Shuffle final result
        )
    else:
        # Sem estratificação se coluna alvo não existir
        sampled = data.sample(
            n=max_samples,
            random_state=random_state
        )

    logger.info(f"Amostra criada com {len(sampled)} linhas")

    return sampled


def create_explainer(model, X: pd.DataFrame) -> object:
    """
    Cria um objeto SHAP Explainer para o modelo.

    Trata Pipeline sklearn extraindo o estimador final.
    Usa TreeExplainer para modelos baseados em árvores,
    ou KernelExplainer como fallback.

    Args:
        model: Modelo sklearn (Pipeline ou estimador simples).
        X: DataFrame de features usada para inicializar o explainer.

    Returns:
        Objeto shap.Explainer inicializado.

    Raises:
        ValueError: Se o tipo de modelo não for suportado.

    Nota:
        TreeExplainer é mais eficiente para modelos baseados em árvores.
        KernelExplainer é universal mas mais lento.
    """
    # Se for Pipeline, extrai o modelo final
    if hasattr(model, "named_steps"):
        logger.info("Detectado Pipeline. Extraindo estimador final...")
        model_estimator = model.named_steps.get("model", model)
    else:
        model_estimator = model

    logger.info(f"Criando SHAP Explainer para {type(model_estimator).__name__}")

    try:
        # Tenta usar TreeExplainer (mais eficiente)
        explainer = shap.TreeExplainer(model_estimator)
        logger.info("Usando TreeExplainer")
    except Exception:
        # Fallback para KernelExplainer (universal)
        logger.warning(
            "TreeExplainer não disponível, usando KernelExplainer (mais lento)"
        )
        explainer = shap.KernelExplainer(
            model_estimator.predict,
            shap.sample(X, min(100, len(X)))
        )

    return explainer


def generate_shap_values(explainer, X: pd.DataFrame) -> object:
    """
    Gera os valores SHAP para o dataset de entrada.

    Os valores SHAP indicam a contribuição de cada feature
    para a previsão do modelo em cada amostra.

    Args:
        explainer: Objeto SHAP Explainer inicializado.
        X: DataFrame de features.

    Returns:
        Objeto shap.Explanation contendo os valores SHAP.

    Exemplo:
        >>> shap_values = generate_shap_values(explainer, X)
        >>> shap_values.shape
        (100, 25)
    """
    logger.info(f"Gerando valores SHAP para {len(X)} amostras...")

    shap_values = explainer(X)

    logger.info("Valores SHAP gerados com sucesso")

    return shap_values


def plot_summary(
    shap_values,
    X: pd.DataFrame,
    output_path: Optional[Path] = None
) -> Path:
    """
    Gera e salva o gráfico de resumo SHAP (força de cada feature).

    O gráfico mostra como cada feature impacta as previsões do modelo,
    com cores indicando valores altos (vermelho) ou baixos (azul).

    Args:
        shap_values: Objeto shap.Explanation com os valores SHAP.
        X: DataFrame original de features.
        output_path: Caminho para salvar a figura. Se None, usa padrão.

    Returns:
        Path do arquivo salvo.

    Nota:
        Usa tight_layout() e alta resolução (300 DPI) para qualidade.
    """
    output_path = output_path or FIGURES_DIR / f"shap_summary.{FIGURE_FORMAT}"

    logger.info("Gerando gráfico de resumo SHAP...")

    # Cria figura
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="dot", show=False)
    plt.tight_layout()

    # Cria diretório se não existir
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Salva figura
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Gráfico de resumo salvo em: {output_path}")

    return output_path


def plot_feature_importance(
    shap_values,
    output_path: Optional[Path] = None
) -> Path:
    """
    Gera e salva o gráfico de importância de features baseado em SHAP.

    O gráfico mostra as features mais importantes (maior impacto médio absoluto).

    Args:
        shap_values: Objeto shap.Explanation com os valores SHAP.
        output_path: Caminho para salvar a figura. Se None, usa padrão.

    Returns:
        Path do arquivo salvo.

    Nota:
        Usa tight_layout() e alta resolução (300 DPI) para qualidade.
    """
    output_path = output_path or FIGURES_DIR / f"shap_bar.{FIGURE_FORMAT}"

    logger.info("Gerando gráfico de importância de features...")

    # Cria figura
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, plot_type="bar", show=False)
    plt.tight_layout()

    # Cria diretório se não existir
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Salva figura
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Gráfico de importância salvo em: {output_path}")

    return output_path


def main(
    max_samples: int = MAX_SAMPLES,
    model_path: Optional[Path] = None,
    features_path: Optional[Path] = None
) -> None:
    """
    Orquestra a execução completa do pipeline de explainabilidade SHAP.

    Este é o ponto de entrada principal que:
    1. Carrega o modelo treinado
    2. Carrega as features
    3. Realiza amostragem para performance
    4. Cria o explainer SHAP
    5. Gera valores SHAP
    6. Cria e salva visualizações

    Args:
        max_samples: Número máximo de amostras para análise (default: 2000).
        model_path: Caminho customizado do modelo (opcional).
        features_path: Caminho customizado das features (opcional).

    Raises:
        FileNotFoundError: Se arquivos necessários não existirem.
        ValueError: Se dados estiverem corrompidos ou inválidos.

    Exemplo:
        >>> main(max_samples=1000)
        # Gera explicações e salva em reports/figures/
    """
    try:
        logger.info("=" * 70)
        logger.info("INICIANDO ANÁLISE DE EXPLAINABILIDADE COM SHAP")
        logger.info("=" * 70)

        # ====================================================================
        # Carregamento de dados
        # ====================================================================
        model = load_model(model_path)
        data = load_data(features_path)

        # Separar features e target
        X = data.drop(columns=[TARGET_COLUMN])
        logger.info(f"Features separadas: {X.shape[1]} colunas")

        # ====================================================================
        # Amostragem para performance
        # ====================================================================
        data_sampled = sample_data(data, max_samples=max_samples)
        X_sampled = data_sampled.drop(columns=[TARGET_COLUMN])

        # ====================================================================
        # Criar explainer
        # ====================================================================
        explainer = create_explainer(model, X)

        # ====================================================================
        # Gerar valores SHAP
        # ====================================================================
        shap_values = generate_shap_values(explainer, X_sampled)

        # ====================================================================
        # Gerar visualizações
        # ====================================================================
        logger.info("Gerando visualizações...")

        path_summary = plot_summary(shap_values, X_sampled)
        path_importance = plot_feature_importance(shap_values)

        logger.info("=" * 70)
        logger.info("ANÁLISE CONCLUÍDA COM SUCESSO")
        logger.info(f"Resumo SHAP salvo em: {path_summary}")
        logger.info(f"Importância salva em: {path_importance}")
        logger.info("=" * 70)

    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado: {e}")
        raise
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado durante análise SHAP: {e}")
        raise


# ============================================================================
# ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    main()
