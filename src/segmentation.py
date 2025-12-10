import pandas as pd

def create_ab_groups(df, feature, A_val, B_val):
    """Return A/B segmented dataset based on feature values"""
    subset = df[df[feature].isin([A_val, B_val])].copy()
    subset["ab_group"] = subset[feature].map({
        A_val: "A_Control",
        B_val: "B_Test"
    })
    return subset
