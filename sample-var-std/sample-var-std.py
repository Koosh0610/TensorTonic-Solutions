import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    x = np.asarray(x,dtype=float)
    var = np.var(x,ddof=1) #N-1, ddof=1
    std_dev = np.sqrt(var)
    return var,std_dev