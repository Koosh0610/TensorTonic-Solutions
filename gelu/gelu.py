import numpy as np
import math
from scipy import special

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: scalar, list, or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x = np.asarray(x,dtype=float)
    return 0.5 * x * (1 + special.erf(x / np.sqrt(2)))