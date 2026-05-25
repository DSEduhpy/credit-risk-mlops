"""
Root conftest.py — loaded by pytest before any test module.

Responsibilities
----------------
1. Set CREDIT_RISK_ENV=test so the config package resolves TestConfig
   before any src import touches the environment.
2. Provide shared fixtures: synthetic DataFrames, mock models,
   temporary directory trees, and patched MLflow.
3. Register the project root on sys.path so ``import src`` works
   without installing the package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CRITICAL: set the environment BEFORE any src import so config resolves
# to TestConfig throughout the entire test session.
# ---------------------------------------------------------------------------
os.environ.setdefault("CREDIT_RISK_ENV", "test")

# ---------------------------------------------------------------------------
# Ensure the project root is importable regardless of how pytest is invoked.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Standard library / third-party imports (after path is set)
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Synthetic dataset factories
# ---------------------------------------------------------------------------

def _make_raw_dataframe(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic raw loan dataset that mimics the LendingClub schema
    used in the actual pipeline, with realistic column distributions.
    """
    rng = np.random.default_rng(seed)

    n = n_rows
    df = pd.DataFrame(
        {
            "loan_amnt": rng.integers(1_000, 40_000, size=n).astype(float),
            "term": rng.choice([" 36 months", " 60 months"], size=n),
            "int_rate": rng.uniform(5.0, 30.0, size=n),
            "installment": rng.uniform(50.0, 1_500.0, size=n),
            "grade": rng.choice(list("ABCDEFG"), size=n),
            "sub_grade": rng.choice(
                [f"{g}{i}" for g in "ABCDE" for i in range(1, 6)], size=n
            ),
            "emp_length": rng.choice(
                ["< 1 year", "1 year", "2 years", "5 years", "10+ years", None],
                size=n,
            ),
            "home_ownership": rng.choice(
                ["RENT", "OWN", "MORTGAGE", "OTHER"], size=n
            ),
            "annual_inc": rng.uniform(20_000, 200_000, size=n),
            "verification_status": rng.choice(
                ["Verified", "Source Verified", "Not Verified"], size=n
            ),
            "purpose": rng.choice(
                [
                    "debt_consolidation",
                    "credit_card",
                    "home_improvement",
                    "other",
                    "small_business",
                ],
                size=n,
            ),
            "dti": rng.uniform(0.0, 40.0, size=n),
            "delinq_2yrs": rng.integers(0, 5, size=n).astype(float),
            "fico_range_low": rng.integers(600, 780, size=n).astype(float),
            "fico_range_high": rng.integers(605, 785, size=n).astype(float),
            "open_acc": rng.integers(1, 30, size=n).astype(float),
            "pub_rec": rng.integers(0, 3, size=n).astype(float),
            "revol_bal": rng.uniform(0, 50_000, size=n),
            "revol_util": rng.uniform(0.0, 100.0, size=n),
            "total_acc": rng.integers(5, 60, size=n).astype(float),
            "mort_acc": rng.integers(0, 10, size=n).astype(float),
            "pub_rec_bankruptcies": rng.integers(0, 2, size=n).astype(float),
            "loan_status": rng.choice(
                ["Fully Paid", "Charged Off"], size=n, p=[0.80, 0.20]
            ),
        }
    )

    # Inject a realistic missing-value pattern (< 5% per column)
    for col in ["emp_length", "revol_util", "mort_acc", "pub_rec_bankruptcies"]:
        mask = rng.random(size=n) < 0.03
        df.loc[mask, col] = np.nan

    return df


def _make_feature_dataframe(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic feature-engineered dataset matching the schema
    produced by src/processing/feature_engineering.py.
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    df = pd.DataFrame(
        {
            "loan_amnt": rng.uniform(1_000, 40_000, size=n),
            "int_rate": rng.uniform(5.0, 30.0, size=n),
            "installment": rng.uniform(50.0, 1_500.0, size=n),
            "annual_inc": rng.uniform(20_000, 200_000, size=n),
            "dti": rng.uniform(0.0, 40.0, size=n),
            "delinq_2yrs": rng.integers(0, 5, size=n).astype(float),
            "fico_range_low": rng.integers(600, 780, size=n).astype(float),
            "open_acc": rng.integers(1, 30, size=n).astype(float),
            "pub_rec": rng.integers(0, 3, size=n).astype(float),
            "revol_bal": rng.uniform(0, 50_000, size=n),
            "revol_util": rng.uniform(0.0, 100.0, size=n),
            "total_acc": rng.integers(5, 60, size=n).astype(float),
            "mort_acc": rng.integers(0, 10, size=n).astype(float),
            "pub_rec_bankruptcies": rng.integers(0, 2, size=n).astype(float),
            "home_ownership_encoded": rng.integers(0, 4, size=n).astype(float),
            "purpose_encoded": rng.integers(0, 5, size=n).astype(float),
            "loan_amnt_to_income": rng.uniform(0.01, 2.0, size=n),
            "fico_avg": rng.uniform(600.0, 785.0, size=n),
            "default": rng.choice([0, 1], size=n, p=[0.80, 0.20]),
        }
    )
    return df


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def raw_dataframe() -> pd.DataFrame:
    """Full-size synthetic raw dataset (200 rows). Session-scoped for speed."""
    return _make_raw_dataframe(n_rows=200)


@pytest.fixture(scope="session")
def small_raw_dataframe() -> pd.DataFrame:
    """Small synthetic raw dataset for fast unit tests (30 rows)."""
    return _make_raw_dataframe(n_rows=30)


@pytest.fixture(scope="session")
def feature_dataframe() -> pd.DataFrame:
    """Full-size synthetic feature dataset (200 rows). Session-scoped."""
    return _make_feature_dataframe(n_rows=200)


@pytest.fixture(scope="session")
def small_feature_dataframe() -> pd.DataFrame:
    """Small synthetic feature dataset for fast unit tests (30 rows)."""
    return _make_feature_dataframe(n_rows=30)


@pytest.fixture
def feature_matrix(small_feature_dataframe: pd.DataFrame):
    """
    Returns (X, y) tuple from the small feature dataset.
    Function-scoped so each test gets a fresh copy.
    """
    df = small_feature_dataframe.copy()
    y = df.pop("default")
    return df, y


@pytest.fixture(scope="session")
def feature_columns(feature_dataframe: pd.DataFrame) -> list:
    """List of feature column names (excludes target)."""
    return [c for c in feature_dataframe.columns if c != "default"]


@pytest.fixture
def mock_trained_model():
    """
    A MagicMock that quacks like a fitted sklearn estimator.
    predict_proba returns a 2-column array with realistic probabilities.
    """
    model = MagicMock()
    model.predict_proba.side_effect = lambda X: np.column_stack(
        [1 - np.random.default_rng(42).uniform(0.1, 0.9, len(X)),
         np.random.default_rng(42).uniform(0.1, 0.9, len(X))]
    )
    model.predict.side_effect = lambda X: (
        model.predict_proba(X)[:, 1] > 0.5
    ).astype(int)
    model.classes_ = np.array([0, 1])
    return model


@pytest.fixture
def tmp_models_dir(tmp_path: Path) -> Path:
    """Temporary directory that mimics the models/ folder."""
    models = tmp_path / "models"
    models.mkdir()
    return models


@pytest.fixture
def tmp_reports_dir(tmp_path: Path) -> Path:
    """Temporary directory that mimics the reports/ folder."""
    reports = tmp_path / "reports"
    (reports / "figures").mkdir(parents=True)
    (reports / "drift").mkdir(parents=True)
    return reports


@pytest.fixture
def patched_mlflow():
    """
    Patch MLflow tracking so tests never write to mlruns/.
    Yields a MagicMock representing the mlflow module.
    """
    with patch("mlflow.start_run"), \
         patch("mlflow.log_metric"), \
         patch("mlflow.log_param"), \
         patch("mlflow.log_artifact"), \
         patch("mlflow.set_experiment"), \
         patch("mlflow.sklearn.log_model"):
        yield


@pytest.fixture(autouse=True)
def _reset_correlation_id():
    """
    Reset the correlation ID context variable between tests to prevent
    state leaking from one test to the next.
    """
    from src.logger import set_correlation_id
    set_correlation_id("test-run")
    yield
    set_correlation_id("no-correlation-id")