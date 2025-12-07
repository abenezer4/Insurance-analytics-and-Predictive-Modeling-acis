# src/data_loader.py
from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_data(
    filename: str = "MachineLearningRating_v3.txt",
    sep: str | None = None,
    dtype_overrides: dict | None = None,
) -> pd.DataFrame:
    
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(
        path,
        sep=sep,          # e.g., ',' or '|' or '\t'
        engine="python",  # safer when sep is None or complex
    )

    if dtype_overrides:
        df = df.astype(dtype_overrides)

    return df
