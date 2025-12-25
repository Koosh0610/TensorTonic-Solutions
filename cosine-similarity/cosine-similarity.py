import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    if np.linalg.norm(np.array(a)) == 0 or np.linalg.norm(np.array(b)) == 0:
        return 0.0
    else:
        return float(np.dot(np.array(a),np.array(b)) / (np.linalg.norm(a)*np.linalg.norm(b)))