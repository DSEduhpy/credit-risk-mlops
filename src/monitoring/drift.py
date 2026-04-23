from pathlib import Path

import pandas as pd

from src.config import CLEAN_DATA_PATH
from src.logger import get_logger

logger = get_logger(__name__)


def compare_drift(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    report = {}
    numeric_columns = reference.select_dtypes(include=["number"]).columns.intersection(
        candidate.select_dtypes(include=["number"]).columns
    )

    for column in numeric_columns:
        ref_mean = reference[column].mean()
        cand_mean = candidate[column].mean()
        ref_std = reference[column].std()
        cand_std = candidate[column].std()
        drift_score = abs(ref_mean - cand_mean) / (abs(ref_mean) + 1e-9)
        report[column] = {
            "reference_mean": float(ref_mean),
            "candidate_mean": float(cand_mean),
            "reference_std": float(ref_std),
            "candidate_std": float(cand_std),
            "drift_score": float(drift_score),
            "drift_detected": drift_score > 0.1,
        }
    return report


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em: {path}")
    return pd.read_parquet(path)


def run_drift_analysis(candidate_path: Path) -> None:
    reference = load_dataset(CLEAN_DATA_PATH)
    candidate = load_dataset(candidate_path)
    report = compare_drift(reference, candidate)
    logger.info("Resultado do drift", extra={"report": report})
    print(report)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Uso: python src/monitoring/drift.py data/processed/new_data.parquet")

    candidate_file = Path(sys.argv[1])
    run_drift_analysis(candidate_file)
