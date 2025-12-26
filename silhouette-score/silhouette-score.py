import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    # Write code here
    n_samples = X.shape[0]
    clusters = np.unique(labels)
    #broadcasting (N,1,D) and (1,N,D) --> (N,N,D)
    diff = X[:,None,:] - X[None,:,:]
    distances = np.linalg.norm(diff,axis=-1)
    silhouettes = np.zeros(n_samples)
    for i in range(n_samples):
        same_cluster = labels == labels[i]
        same_cluster[i] = False
        if np.any(same_cluster):
            a_i = distances[i, same_cluster].mean()
        else:
            silhouettes[i] = 0.0
            continue
        b_i = np.inf
        for c in clusters:
            if c == labels[i]:
                continue
            other_cluster = labels == c
            b_i = min(b_i, distances[i, other_cluster].mean())
        silhouettes[i] = (b_i - a_i) / max(a_i, b_i)
    return silhouettes.mean()