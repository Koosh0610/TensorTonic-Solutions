import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.atleast_1d(np.asarray(x, dtype=float))
    a = np.exp(x)
    b = np.exp(-x)
    return (a-b) / (a+b)