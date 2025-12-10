import numpy as np
from scipy.stats.contingency import association

def cramers_v(cont_table):
    return association(cont_table, method="cramer")

def cohens_d(g1, g2):
    diff = g1.mean() - g2.mean()
    pooled = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
    return diff / pooled if pooled > 0 else 0
