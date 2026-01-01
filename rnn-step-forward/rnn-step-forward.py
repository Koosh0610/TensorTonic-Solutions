import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    x_t = np.atleast_1d(np.asarray(x_t,dtype=float))
    h_prev = np.atleast_1d(np.asarray(h_prev,dtype=float))
    Wx = np.atleast_1d(np.asarray(Wx,dtype=float))
    Wh = np.atleast_1d(np.asarray(Wh,dtype=float))
    b = np.atleast_1d(np.asarray(b,dtype=float))
    return np.tanh(x_t@Wx + h_prev@Wh + b)
