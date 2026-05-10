# Credit Risk MLOps Pipeline

## Overview

This project addresses the critical problem of automated credit decision-making for consumer portfolios. The solution supports large-scale credit risk assessment, focusing on maximizing financial returns and minimizing operational losses.

Instead of optimizing generic classification metrics, the model is calibrated for real business impact:
- Cost of default per client: `10000`
- Recoverable revenue per approved client: `1000`

The estimated project result is an incremental financial gain of approximately `+23 million`, maintaining a technical balance between AUC, precision, and recall.

## Business Problem

Credit risk assessment involves balancing two key financial metrics:
- **Default cost**: High cost when approving bad payers (false negatives)
- **Revenue opportunity**: Lost profit when rejecting good payers (false positives)

The goal is to maximize net financial return by optimizing the decision threshold based on actual business costs, not statistical metrics.

## Architecture

The system follows a modular architecture with clear separation of concerns:

- **ingestion**: Captures and consolidates source data into parquet format
- **processing**: Data cleaning, missing value handling, and preprocessing
- **feature_engineering**: Categorical transformation and feature generation
- **modeling**: Training with experiment tracking, artifact production, and threshold optimization
- **api**: FastAPI serving layer for real-time inference
- **monitoring**: Drift detection and data quality monitoring
- **explainability**: SHAP-based model explanations

The pipeline is orchestrated by DVC, ensuring reproducibility of data, logic, and artifacts.

```
credit-risk-mlops/
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── docs/
├── models/
├── reports/
├── src/
│   ├── ingestion/
│   ├── processing/
│   ├── modeling/
│   ├── evaluation/
│   ├── api/
│   ├── monitoring/
│   └── explainability/
├── tests/
├── .github/workflows/
├── dvc.yaml
├── Dockerfile
├── requirements.txt
├── README.md
└── tasklist.md
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Language | Python 3.x | Core development |
| Data Processing | Pandas, PyArrow | Data manipulation and columnar storage |
| Machine Learning | Scikit-learn, XGBoost, LightGBM, CatBoost | Modeling and algorithms |
| Experiment Tracking | MLflow | Model versioning and metrics |
| Data Versioning | DVC | Pipeline orchestration and data versioning |
| API | FastAPI | Real-time inference serving |
| Explainability | SHAP | Model interpretability |
| Containerization | Docker | Environment portability |
| Visualization | Matplotlib | Charts and plots |

## Pipeline

### Data Pipeline
- **ingest**: Validates CSV source and creates `data/raw/data.parquet`
- **processing**: Cleans data, handles missing values, applies transformations
- **features**: Generates model-ready features with one-hot encoding

### Training Pipeline
- **train**: Trains multi-model benchmark with MLflow tracking

## Modeling

The project implements a multi-model benchmark for binary classification:

- **Logistic Regression**: Baseline linear model
- **XGBoost**: Gradient boosting with tree-based optimization
- **LightGBM**: Microsoft gradient boosting framework
- **CatBoost**: Yandex categorical feature handling

All models are trained with:
- Stratified train/test splits to handle class imbalance
- Class weighting to mitigate bias
- Comprehensive metric logging (AUC, precision, recall, financial impact)

## Business Optimization

### Threshold Calibration
Instead of using default 0.5 threshold, the system optimizes based on business costs:
- Default cost: 10,000 per bad approval
- Revenue: 1,000 per good approval

The optimal threshold maximizes net profit by finding the right balance between precision (avoiding bad approvals) and recall (capturing defaults).

## Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| AUC | ~0.748 | Good separation between good/bad payers |
| Precision | ~0.17 | Accuracy of positive predictions in imbalanced context |
| Recall | ~0.64 | Ability to capture actual defaulters |
| Financial Impact | +23M | Estimated incremental profit |

## MLflow Tracking

Experiments are tracked with:
- Model parameters and hyperparameters
- Performance metrics (AUC, precision, recall)
- Business metrics (profit, loss)
- Model artifacts and versions

Access tracking UI with: `mlflow ui --backend-store-uri ./mlruns`

## API

FastAPI-based inference service with endpoint:

```bash
POST /predict
Content-Type: application/json

{
  "features": {
    "AMT_INCOME_TOTAL": 202500.0,
    "DAYS_BIRTH": -12012,
    "CODE_GENDER": "F"
  }
}
```

## Explainability

SHAP-based explanations for model interpretability. See [docs/explainability.md](docs/explainability.md) for details.

## Monitoring

Drift detection monitors:
- Feature distribution changes
- Prediction distribution shifts
- Data quality metrics

## Run Locally

### Prerequisites
- Python 3.8+
- Git

### Setup
```bash
# Clone repository
git clone <repository-url>
cd credit-risk-mlops

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Run DVC pipeline
dvc repro

# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Start API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Test inference
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"AMT_INCOME_TOTAL": 202500.0, "DAYS_BIRTH": -12012, "CODE_GENDER": "F"}}'
```

## Roadmap

- [x] Multi-model benchmark (Logistic, XGBoost, LightGBM, CatBoost)
- [x] SHAP explainability implementation
- [x] Business-oriented threshold optimization
- [ ] GitHub Actions CI/CD pipeline
- [ ] Cloud deployment (AWS/GCP) with containers
- [ ] Advanced monitoring and alerting
- [ ] Automated testing suite
- [ ] Feature store implementation
- [ ] Model A/B testing framework
