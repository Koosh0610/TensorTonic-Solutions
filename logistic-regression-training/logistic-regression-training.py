import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    N , features = X.shape
    W = np.zeros(features)
    b = 0.0
    for _ in range(steps):
        z = X@W + b
        grad_w = X.T@(_sigmoid(z)-y) / N
        grad_b = np.mean(_sigmoid(z)-y)
        W = W - lr*grad_w
        b = b - lr*grad_b
    return (W,b)