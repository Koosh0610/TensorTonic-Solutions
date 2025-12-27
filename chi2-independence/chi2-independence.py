import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Write code here
    C = np.asarray(C,dtype=float)
    row_sums = C.sum(axis=1,keepdims=True)
    col_sums = C.sum(axis=0,keepdims=True)
    total = C.sum()
    E = (row_sums*col_sums)/ total
    chi_square = np.sum((C-E)**2/E)
    return chi_square,E