# 💳 Credit Risk MLOps Pipeline

Pipeline completo de Machine Learning para análise e gerenciamento de risco de crédito, desenvolvido com foco em boas práticas de Engenharia de Dados, Ciência de Dados, MLOps e Engenharia de Software.

---

## 🎯 Objetivo

Este projeto implementa um pipeline end-to-end para previsão de inadimplência em operações de crédito.

O sistema permite:

* Ingestão e validação de dados
* Processamento e limpeza
* Engenharia de atributos
* Treinamento de modelos de Machine Learning
* Avaliação técnica e financeira
* Versionamento de dados e modelos
* Monitoramento de qualidade
* Disponibilização via API REST

O foco principal é gerar valor de negócio através da redução de perdas financeiras causadas por inadimplência.

---

# 🏗️ Arquitetura

```text
Raw Data
    │
    ▼
┌───────────────┐
│   Ingestion   │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Validation    │
│ Schema Check  │
│ Data Quality  │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Processing    │
│ Cleaning      │
│ Missing Values│
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Feature Eng.  │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Modeling      │
│ MLflow        │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Evaluation    │
│ Business KPI  │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ FastAPI       │
│ Inference API │
└───────────────┘
```

---

# 📂 Estrutura do Projeto

```text
credit-risk-mlops/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── models/
│
├── reports/
│
├── src/
│   ├── ingestion/
│   ├── processing/
│   ├── validation/
│   ├── modeling/
│   ├── evaluation/
│   ├── api/
│   ├── monitoring/
│   └── explainability/
│
├── tests/
│
├── .github/
│   └── workflows/
│
├── dvc.yaml
├── requirements.txt
├── Dockerfile
├── README.md
└── tasklist.md
```

---

# 🛠️ Tecnologias

## Linguagem

* Python 3.12+

## Manipulação de Dados

* Pandas
* NumPy
* PyArrow

## Machine Learning

* Scikit-Learn
* XGBoost
* LightGBM
* CatBoost

## MLOps

* MLflow
* DVC

## API

* FastAPI
* Uvicorn
* Pydantic

## Qualidade

* Pytest
* Ruff

## Infraestrutura

* Docker
* GitHub Actions

---

# 📊 Funcionalidades Implementadas

## Engenharia de Dados

* Ingestão de dados
* Conversão para Parquet
* Pipeline reprodutível
* Validação de schema

## Qualidade de Dados

* Verificação de colunas obrigatórias
* Validação de tipos
* Controle de dados faltantes
* Detecção de drift de schema

## Ciência de Dados

* Feature Engineering
* Seleção de atributos
* Treinamento de modelos
* Avaliação de desempenho

## Métricas de Negócio

* Receita estimada
* Custo de inadimplência
* Otimização baseada em lucro

## MLOps

* Rastreamento de experimentos
* Versionamento de datasets
* Versionamento de modelos
* Testes automatizados

---

# 🧪 Testes

Executar todos os testes:

```bash
pytest tests -v
```

Executar testes específicos:

```bash
pytest tests/validation -v
```

Com cobertura:

```bash
pytest --cov=src
```

---

# 🚀 Como Executar

## 1. Clonar repositório

```bash
git clone https://github.com/seu-usuario/credit-risk-mlops.git

cd credit-risk-mlops
```

## 2. Criar ambiente virtual

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

## 3. Instalar dependências

```bash
pip install -r requirements.txt
```

## 4. Executar pipeline

```bash
dvc repro
```

---

# 📈 MLflow

Iniciar interface:

```bash
mlflow ui
```

Acessar:

```text
http://localhost:5000
```

---

# 🌐 API

Executar:

```bash
uvicorn src.api.app:app --reload
```

Documentação:

```text
http://localhost:8000/docs
```

---

# 🔄 CI/CD

O projeto utiliza GitHub Actions para:

* Execução automática de testes
* Validação de código
* Verificação de qualidade

---

# 📌 Roadmap

* [x] Ingestão de dados
* [x] Processamento
* [x] Feature Engineering
* [x] Validação de Schema
* [x] Validação de Qualidade
* [x] Testes Automatizados
* [x] Métricas de Negócio
* [ ] Treinamento Final dos Modelos
* [ ] API Completa de Inferência
* [ ] Monitoramento de Produção
* [ ] Explainability com SHAP
* [ ] Deploy em Cloud

---

# 👨‍💻 Autor

Eduardo

Projeto desenvolvido para demonstrar competências em:

* Engenharia de Dados
* Ciência de Dados
* Machine Learning
* MLOps
* Engenharia de Software
