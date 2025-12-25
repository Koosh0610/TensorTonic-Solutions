import numpy.polynomial as poly  
def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    p = poly.Polynomial([c,b,a])
    p_deriv = p.deriv()
    for _ in range(steps):
        x0 = x0 - lr*p_deriv(x0)
    return (x0)