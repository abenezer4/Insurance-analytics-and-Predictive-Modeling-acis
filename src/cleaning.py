
import pandas as pd
import numpy as np
from pathlib import Path

# ===============================
# 1. COLUMN CLEANING
# ===============================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names:
    - Strips spaces
    - Converts to lowercase
    - Replaces spaces, hyphens, slashes with underscores
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )
    if "totalclaims" not in df.columns:
        print(f"[] ❌ Column 'totalclaims' not found.")
        return

    # Ensure we have numeric dtype (NaN-safe)
    s = pd.to_numeric(df["totalclaims"], errors="coerce")

    non_zero = (s != 0).sum()
    total    = s.notna().sum()          # ignore NaNs in denominator
    pct      = (non_zero / total * 100) if total else 0.0

    print(f"[] Non-zero totalclaims: {non_zero:,} / {total:,}  ({pct:.2f}%)")
    return df


# ===============================
# 2. DATA TYPE FIXING
# ===============================

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts:
    - Date columns → datetime
    - Numeric columns → numeric
    - Categorical columns → category
    """

    # Identify date columns
    date_cols = [col for col in df.columns if "month" in col or "date" in col]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Convert obvious numeric columns
    numeric_like = [
        "totalpremium",
        "totalclaims",
        "suminsured",
        "cubiccapacity",
        "kilowatts",
        "customvalueestimate",
    ]

    for col in numeric_like:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    
    categorical_like = [
        "province",
        "gender",
        "vehicletype",
        "maritalstatus",
        "country",
        "postalcode",
        "product",
        "covertype",
        "covercategory",
    ]

    for col in categorical_like:
        if col in df.columns:
            df[col] = df[col].astype("category")
    if "totalclaims" not in df.columns:
        print(f"[] ❌ Column 'totalclaims' not found.")
        return

    # Ensure we have numeric dtype (NaN-safe)
    s = pd.to_numeric(df["totalclaims"], errors="coerce")

    non_zero = (s != 0).sum()
    total    = s.notna().sum()          # ignore NaNs in denominator
    pct      = (non_zero / total * 100) if total else 0.0

    print(f"[] Non-zero totalclaims: {non_zero:,} / {total:,}  ({pct:.2f}%)")

    return df


# ===============================
# 3. MISSING VALUES
# ===============================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drops columns with >60% missing
    - Fills numeric with median
    - Fills categorical with mode
    """

    missing_pct = df.isna().mean()

    cols_to_drop = missing_pct[missing_pct > 0.60].index
    df = df.drop(columns=cols_to_drop)

    # Numeric
    for col in df.select_dtypes(include="number"):
        df[col] = df[col].fillna(df[col].median())

    # Categorical
    for col in df.select_dtypes(include="category"):
        try:
            df[col] = df[col].fillna(df[col].mode()[0])
        except Exception:
            df[col] = df[col].fillna("Unknown")
    if "totalclaims" not in df.columns:
        print(f"[] ❌ Column 'totalclaims' not found.")
        return

    # Ensure we have numeric dtype (NaN-safe)
    s = pd.to_numeric(df["totalclaims"], errors="coerce")

    non_zero = (s != 0).sum()
    total    = s.notna().sum()          # ignore NaNs in denominator
    pct      = (non_zero / total * 100) if total else 0.0

    print(f"[] Non-zero totalclaims: {non_zero:,} / {total:,}  ({pct:.2f}%)")

    return df


# ===============================
# 4. FEATURE ENGINEERING
# ===============================

def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - lossratio = totalclaims / totalpremium
    - has_claim = 1 if totalclaims > 0
    """

    if "totalclaims" in df.columns and "totalpremium" in df.columns:
        df["lossratio"] = df["totalclaims"] / df["totalpremium"]
        df["lossratio"] = df["lossratio"].replace([np.inf, -np.inf], np.nan)
        df["lossratio"] = df["lossratio"].fillna(0)

    if "totalclaims" in df.columns:
        df["has_claim"] = (df["totalclaims"] > 0).astype(int)
    if "totalclaims" not in df.columns:
        print(f"[] ❌ Column 'totalclaims' not found.")
        return

    # Ensure we have numeric dtype (NaN-safe)
    s = pd.to_numeric(df["totalclaims"], errors="coerce")

    non_zero = (s != 0).sum()
    total    = s.notna().sum()          # ignore NaNs in denominator
    pct      = (non_zero / total * 100) if total else 0.0

    print(f"[] Non-zero totalclaims: {non_zero:,} / {total:,}  ({pct:.2f}%)")

    return df


# ===============================
# 5. OUTLIER HANDLING
# ===============================

def winsorize_series(series: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    """Caps extreme values to reduce outlier influence."""
    if series.dtype != "float" and series.dtype != "int":
        return series

    q_low, q_high = series.quantile([lower, upper])
    
    return np.clip(series, q_low, q_high)


def apply_outlier_treatment(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        df[col] = winsorize_series(df[col])
    if "totalclaims" not in df.columns:
        print(f"[] ❌ Column 'totalclaims' not found.")
        return

    # Ensure we have numeric dtype (NaN-safe)
    s = pd.to_numeric(df["totalclaims"], errors="coerce")

    non_zero = (s != 0).sum()
    total    = s.notna().sum()          # ignore NaNs in denominator
    pct      = (non_zero / total * 100) if total else 0.0

    print(f"[] Non-zero totalclaims: {non_zero:,} / {total:,}  ({pct:.2f}%)")
    return df


# ===============================
# 6. MAIN CLEANING PIPELINE
# ===============================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Clean column names
    2. Convert data types
    3. Handle missing values
    4. Add derived fields (loss ratio, has_claim)
    5. Apply outlier treatment
    """
    df = clean_column_names(df)
    df = convert_data_types(df)
    df = handle_missing_values(df)
    df = add_derived_fields(df)
    

    return df


