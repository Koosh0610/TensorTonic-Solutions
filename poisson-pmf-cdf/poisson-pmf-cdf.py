import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    k = int(k)
    if k == 0:
        log_fact = 0
    else:
        log_fact = np.sum(np.log(np.arange(1, k+1)))
    
    pmf = np.exp(k * np.log(lam) - lam - log_fact)
    
    cdf = 0.0
    for i in range(k+1):
        if i == 0:
            log_fact_i = 0
        else:
            log_fact_i = np.sum(np.log(np.arange(1, i+1)))
        cdf += np.exp(i * np.log(lam) - lam - log_fact_i)
    
    return float(pmf), float(cdf)