import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    y_left = np.asarray(y_left,dtype=float)
    y_right = np.asarray(y_right,dtype=float)
    unique_clases_left,counts_left = np.unique(y_left,return_counts=True)
    unique_clases_right,counts_right = np.unique(y_right,return_counts=True)

    total = np.sum(counts_left) + np.sum(counts_right)
    if total == 0:
        return 0.0  

    if counts_left.size == 0:
        gini_l = 0.0
    else:
        gini_l = 1 - np.sum((counts_left / np.sum(counts_left)) ** 2)
    if counts_right.size == 0:
        gini_r = 0.0
    else:
        gini_r = 1 - np.sum((counts_right / np.sum(counts_right)) ** 2)

    gini_split = (1/(np.sum(counts_left)+np.sum(counts_right)))*(np.sum(counts_left)*gini_l + np.sum(counts_right)*gini_r)

    return gini_split