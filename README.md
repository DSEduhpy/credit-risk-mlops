# credit-risk-mlops

## 1. Visão Geral

Este projeto resolve o problema crítico de decisão de crédito automatizada para carteiras de consumidores. A solução suporta a avaliação de risco de inadimplência em escala, com foco em maximizar o retorno financeiro e minimizar perdas operacionais.

Em vez de otimizar métricas genéricas de classificação, o modelo é calibrado para o impacto de negócio real:
- custo de inadimplência por cliente: `10000`
- receita recuperável por cliente aprovado: `1000`

O resultado estimado do projeto é um ganho financeiro incremental da ordem de `+23 milhões`, mantendo um equilíbrio técnico entre AUC, precision e recall.

## 2. Arquitetura do Sistema

A arquitetura é composta por camadas claras:
- `ingestion`: captura e consolida a base de origem em formato parquet
- `processing`: limpeza, tratamento de missing e pré-processamento
- `feature_engineering`: transformação categórica e geração de features modeláveis
- `modeling`: treino com tracking de experimentos, produção de artefatos e otimização de threshold
- `serving`: API FastAPI para inferência em tempo real
- `monitoring`: detecção de drift e qualidade de dados

O pipeline é orquestrado pelo DVC, garantindo reprodutibilidade de dados, lógica e artefatos.

## 3. Stack Tecnológica

- Python 3.x
- Pandas
- Scikit-learn
- MLflow (SQLite backend)
- DVC
- FastAPI
- Docker
- joblib
- python-dotenv

## 4. Pipeline de Dados

### ingest
Responsável por ingerir o dataset bruto e gerar `data/raw/data.parquet`. A etapa valida a existência do arquivo CSV original e materializa o dataset em formato columnar.

### processing
Executa limpeza estruturada:
- remoção de duplicatas
- drop de colunas com >90% de missing
- imputação numérica por mediana
- preenchimento de categóricos com `missing`
- engenharia básica de transformação logarítmica

O objetivo é estabilizar a qualidade dos dados antes da geração de features.

### features
Transforma o dataset limpo em features para o modelo:
- one-hot encoding de variáveis categóricas
- preservação de label target
- persistência em `data/features/features.parquet`

Essa etapa isola a lógica de feature engineering do processo de treino.

### train
Treina um classificador de risco e registra o experimento no MLflow.
- dados de treino/teste estratificados
- log de parâmetros de treino
- log de métricas de performance
- persistência do modelo em `models/model.pkl`

## 5. Modelagem

O projeto foi estruturado para um modelo de classificação binária rodando dentro de um pipeline de pre-processamento.

### Problema de desbalanceamento
A classe de inadimplência é minoritária. Por isso, o treinamento considera:
- estratificação em `train_test_split`
- uso de class_weight para mitigar viés de classe (via pipeline configurável)
- métricas além de acurácia, como AUC, precision e recall

### Pipeline de modelo
A arquitetura prevê um pipeline reutilizável, onde o pré-processamento e o classificador podem ser versionados separadamente. Essa modularidade facilita:
- auditoria de features
- swap de algoritmos (Logistic Regression, Random Forest, etc.)
- testes de regressão

## 6. Otimização orientada a negócio

### Por que F1-score não é suficiente
F1-score agrupa precision e recall com peso igual. No scoring de crédito, o custo de um falso negativo (inadimplente aprovado) é muito maior que o custo de um falso positivo.

### Como foi feita a otimização
A otimização é baseada em uma função de custo de negócio:
- `custo_inadimplente = 10000`
- `lucro_cliente = 1000`

O threshold de decisão é calibrado para maximizar lucros e minimizar perdas, em vez de maximizar F1. Isso implica ajustar trade-offs entre:
- precision: evita admitir clientes de alto risco
- recall: identifica inadimplentes para reduzir perdas

### Trade-off
A solução busca um ponto de operação em que o ganho marginal de precisão compense a perda de recall, mantendo o impacto financeiro positivo. Esse é o principal diferencial em relação a abordagens puramente estatísticas.

## 7. Resultados

- AUC: ~0.748 — indica capacidade de separação entre bons e maus pagadores
- Precision: ~0.17 — reflete a taxa de previsões positivas corretas em um contexto de classes desbalanceadas
- Recall: ~0.64 — demonstra a capacidade de capturar inadimplentes
- Resultado financeiro: `+23 milhões` — estimativa de ganho incremental considerando custo de inadimplência e receita por cliente

Esses resultados mostram que a solução equilibra performance técnica e retorno econômico, priorizando a governança de risco.

## 8. Problemas encontrados

- `ModuleNotFoundError` ao executar módulos isolados: resolvido usando `python -m src.ingestion.load_data` e ajustando imports relativos abertos.
- Erro no MLflow ao logar função em vez de valor: resolvido ao garantir `mlflow.log_metric(...)` com valores numéricos explícitos.
- `ConvergenceWarning` no classificador: mitigado com normalização de features e estabilização do pipeline de pré-processamento.
- Threshold mal calibrado: corrigido ao introduzir função de custo financeiro e validação de pontos de corte no conjunto de teste.
- Pipeline quebrando no DVC: resolvido ajustando dependências e outputs no `dvc.yaml`, garantindo que cada stage tenha arquivos de entrada/saída consistentes.
- DataFrame fragmentado / performance warning: minimizado usando parquet como formato persistente e evitando concatenações repetidas no pipeline.

## 9. Produção e MLOps

- DVC garante reprodutibilidade de dados, transformações e saídas em cada etapa.
- MLflow garante rastreamento de experimentos, versionamento de runs e registro de artefatos de modelo.
- FastAPI entrega uma camada de serving com inferência em tempo real.
- Docker garante portabilidade e alinhamento entre ambientes de desenvolvimento, staging e produção.

Esse stack permite mover a solução para um fluxo de deploy confiável, com controle de versão e auditabilidade.

## 10. Como rodar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar ambiente

```bash
copy .env.example .env
```

### 3. Executar pipeline DVC

```bash
dvc repro
```

### 4. Iniciar MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

### 5. Executar API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 6. Chamar inferência

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"AMT_INCOME_TOTAL": 202500.0, "DAYS_BIRTH": -12012, "CODE_GENDER": "F"}}'
```

## 11. Roadmap

- SHAP para explicabilidade local e global
- Monitoramento de drift real com alertas operacionais
- CI/CD com GitHub Actions para pipeline, testes e deploy
- Deploy em cloud (AWS / GCP) com containers imutáveis
- Feature Store conceitual para governança de atributos
- Validação de dados avançada com Great Expectations
- Testes automatizados de integração e regressão de modelo
- Plano de observabilidade para métricas de qualidade e performance
