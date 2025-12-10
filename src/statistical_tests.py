from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from effects import cramers_v, cohens_d
import pandas as pd
def claim_frequency_test(df):
    cont = pd.crosstab(df['ab_group'], df['has_claim'])
    chi2, p, _, _ = chi2_contingency(cont)
    return chi2, p, cramers_v(cont)

def claim_severity_test(df):
    A = df[(df['ab_group'] == 'A_Control') & (df['totalclaims'] > 0)]['totalclaims']
    B = df[(df['ab_group'] == 'B_Test') & (df['totalclaims'] > 0)]['totalclaims']
    t, p = ttest_ind(A, B, equal_var=False)
    return t, p, cohens_d(A, B)

def margin_test(df):
    A = df[df['ab_group'] == 'A_Control']['margin']
    B = df[df['ab_group'] == 'B_Test']['margin']
    f, p = f_oneway(A, B)
    return f, p
