import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here
    if norm_type not in {'l1', 'l2', 'max'}:
        return None
    matrix = np.asarray(matrix,dtype=float)
    if matrix.ndim != 2:
            return None
    if norm_type=='l1':
        norm = np.linalg.norm(matrix,axis=axis,ord=1,keepdims=True)
    elif norm_type=='l2':
        norm =  np.linalg.norm(matrix,axis=axis,keepdims=True)
    else:
        norm = np.linalg.norm(matrix,axis=axis,ord=np.inf,keepdims=True)
    norm[norm == 0] = 1
    return matrix / norm
