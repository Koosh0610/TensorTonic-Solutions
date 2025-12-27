import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    #return np.sum(0.5*(tpr[1:]+tpr[:-1])*np.diff(fpr))
    return np.trapezoid(tpr,fpr)