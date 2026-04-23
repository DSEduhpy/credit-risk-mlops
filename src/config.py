import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT.parent / os.getenv("DATA_PATH", "data")
RAW_CSV_PATH = PROJECT_ROOT.parent / "application_train.csv"
RAW_DATA_PATH = DATA_ROOT / "raw" / "data.parquet"
CLEAN_DATA_PATH = DATA_ROOT / "processed" / "clean.parquet"
FEATURES_PATH = DATA_ROOT / "features" / "features.parquet"
MODEL_PATH = PROJECT_ROOT.parent / os.getenv("MODEL_PATH", "models/model.pkl")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT.parent / "mlruns"))

TARGET_COLUMN = "TARGET"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 6
