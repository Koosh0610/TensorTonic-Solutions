import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute k nearest neighbor indices for each test sample.
    If k > number of training samples, pad with -1.
    """

    # Normalize shapes (supports 1D and multi-D features)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    if X_train.ndim == 1:
        X_train = X_train[:, None]
    if X_test.ndim == 1:
        X_test = X_test[:, None]

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Compute pairwise distances
    distances = np.linalg.norm(
        X_test[:, None, :] - X_train[None, :, :],
        axis=-1
    )

    # Actual number of neighbors available
    k_eff = min(k, n_train)

    # Get indices of nearest neighbors
    nn_idx = np.argsort(distances, axis=1)[:, :k_eff]

    # Prepare padded output
    neighbors = np.full((n_test, k), -1, dtype=int)
    neighbors[:, :k_eff] = nn_idx

    return neighbors
