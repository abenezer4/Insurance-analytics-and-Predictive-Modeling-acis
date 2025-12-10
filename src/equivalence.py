import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from effects import cramers_v, cohens_d

def check_equivalence(df, group_col, vars_list):
    """
    Check confounding balance using t-tests + chi-square tests.
    Returns a dictionary of results.
    """
    results = {}

    for var in vars_list:
        if var not in df.columns:
            continue

        if df[var].dtype == "object" or df[var].nunique() <= 10:
            cont = pd.crosstab(df[group_col], df[var])
            chi2, p, _, _ = chi2_contingency(cont)
            results[var] = {
                "type": "categorical",
                "p_value": p,
                "effect_size": cramers_v(cont)
            }
        else:
            gA = df[df[group_col] == "A_Control"][var].dropna()
            gB = df[df[group_col] == "B_Test"][var].dropna()

            t, p = ttest_ind(gA, gB, equal_var=False)
            results[var] = {
                "type": "numeric",
                "p_value": p,
                "effect_size": cohens_d(gA, gB)
            }

    return results
