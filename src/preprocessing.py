import pandas as pd


def preprocess_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-processing pipeline for EDA + Statistical Testing.
    Steps:
    1. Remove duplicates
    2. Ensure correct numeric types
    3. Encode categorical variables (optional for modeling)
    4. Remove unrealistic values
    """

  
    df = df.drop_duplicates()

  
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

   
    for col in ["totalclaims", "totalpremium", "suminsured"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)



    if "totalpremium" in df.columns and "totalclaims" in df.columns:
        df["margin"] = df["totalpremium"] - df["totalclaims"]

   
    if "lossratio" in df.columns:
        df["risk_bucket"] = pd.cut(
            df["lossratio"],
            bins=[-0.01, 0.3, 0.6, 1.0, 10],
            labels=["low", "moderate", "high", "very_high"]
        )

    return df
