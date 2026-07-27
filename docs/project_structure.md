# Estrutura do Projeto

## Visão geral

O repositório está organizado em camadas funcionais que isolam responsabilidades e mantêm o código modular. Cada pacote ou diretório atende a uma etapa específica do pipeline de crédito.

## Diretórios principais

- `data/`
  - `raw/`: dados de origem importados e versionados.
  - `processed/`: dados limpos e transformados prontos para feature engineering.
  - `features/`: datasets de features prontos para modelagem.

- `docs/`
  - Documentação de arquitetura, estrutura e análises técnicas.

- `models/`
  - Artefatos de modelo serializados usados pela API e explainability.

- `reports/`
  - Resultados de avaliação, drift, qualidade e figuras geradas.

- `src/`
  - Código fonte do pipeline, API e utilitários.

- `tests/`
  - Suíte de testes unitários e de integração.

## Pacotes `src/`

### `src/config`

Responsabilidade: configuração baseada em ambiente.

Arquivos:
- `__init__.py`: resolve `CREDIT_RISK_ENV` e exporta `settings`.
- `base.py`: definição de classes de configuração central.
- `dev.py`: valores para ambiente de desenvolvimento.
- `prod.py`: valores para ambiente de produção.
- `test.py`: valores para ambiente de teste.

Quem utiliza:
- todos os módulos da aplicação.
- testes usam `TestConfig` para garantir isolamento.

### `src/logger`

Responsabilidade: logging estruturado e centralizado.

Funcionalidades:
- formatação JSON e humana
- logger por estágio do pipeline
- inferência e drift logging
- timer de estágios

Quem utiliza:
- `src/api`, `src/modeling`, `src/monitoring`, `src/validation`, `src/explainability`, `src/processing`.

### `src/ingestion`

Responsabilidade: leitura e ingestão de dados brutos.

Arquivos:
- `load_data.py`

Quem utiliza:
- DVC e pipelines de ingestão.

### `src/processing`

Responsabilidade: limpeza and transformação de dados.

Arquivos:
- `cleaning.py`
- `feature_engineering.py`

Quem utiliza:
- `src/modeling` para treino.
- `tests/processing` para validação.

### `src/modeling`

Responsabilidade: construção, treinamento e benchmark de modelos.

Arquivos:
- `data.py`: carregamento de features e split.
- `train.py`: treinamento e registro no MLflow.
- `models/`: construtores de modelos específicos.

Quem utiliza:
- `src/api` para inferência.
- `src/explainability` para explicações.

### `src/api`

Responsabilidade: API de inferência em tempo real.

Arquivos:
- `app.py`

Quem utiliza:
- clientes externos.
- validação de modelo em produção.

### `src/monitoring`

Responsabilidade: detecção de drift e relatórios.

Arquivos:
- `drift.py`

Quem utiliza:
- pipeline de monitoramento.
- dashboards e alertas.

### `src/validation`

Responsabilidade: validação de schema, qualidade e negócios.

Arquivos:
- `expectations.py`
- `quality.py`
- `schema.py`
- `validator.py`

Quem utiliza:
- pipelines de ingestão e produção.
- testes de qualidade.

### `src/explainability`

Responsabilidade: explicabilidade do modelo via SHAP.

Arquivos:
- `explain.py`

Quem utiliza:
- analistas de dados
- equipes de compliance
- relatórios de explicabilidade

## Fluxo de dependências

- `src/config` é base comum para `src/api`, `src/modeling`, `src/validation`, `src/monitoring`, `src/explainability`, `src/processing`.
- `src/logger` é usado por quase todos os módulos.
- `src/modeling` depende de `src/evaluation`.
- `src/validation` depende de `src/processing` apenas indiretamente via dados transformados.

## Testes

- `tests/test_config.py`: valida exportações compatíveis e a resolução de `TestConfig`.
- `tests/modeling/test_data.py`: valida split e schema de dados.
- `tests/processing/test_cleaning.py`: valida limpeza.
- `tests/processing/test_feature_engineering.py`: valida feature engineering.
- `tests/validation/test_schema_and_quality.py`: valida schema e qualidade.
- `tests/api/test_app.py`: valida contratos da API.
- `tests/integration/test_pipeline_integration.py`: valida fluxo end-to-end.

## Arquivos de raiz

- `Dockerfile`: define container da aplicação.
- `dvc.yaml` e `dvc.lock`: pipeline e versionamento de artefatos.
- `requirements/`: dependências separadas por ambiente.
  - `requirements/base.txt`: dependências de execução.
  - `requirements/dev.txt`: dependências de desenvolvimento e testes.
  - `requirements/prod.txt`: dependências de produção.
- `requirements.txt`: arquivo de compatibilidade legada que referencia `requirements/dev.txt`.
- `pytest.ini`: configuração do pytest.
- `README.md`: visão geral do projeto.
- `tasklist.md`: roadmap de entregas.

## Observações de compatibilidade

- O projeto mantém contratos públicos no pacote `src/`.
- Alterações em `src/config/__init__.py` e `src/logger.py` não quebram importações existentes.
- A API pública permanece compatível com o atual endpoint `/predict`.
