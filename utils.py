from typing import Callable
import numpy as np

def calc_error(
        y: np.ndarray,
        t: np.ndarray,
        g: Callable[[float], np.ndarray] = None,
        f: Callable[[float], np.ndarray] = None,
        h: Callable[[float], np.ndarray] = None
    ):
    N = y.shape[0]
    if g is None:
        g = lambda t: np.zeros((N,N))
    if f is None:
        f = lambda t: np.zeros((N,N))
    if h is None:
        h = lambda t: np.zeros(N)
    
    ydot = np.gradient(y, t, axis=1)
    yddot = np.gradient(ydot, t, axis=1)
    err = [yddot[:,i] + f(ti) @ ydot[:,i] + g(ti) @ y[:,i] + h(ti) for i,ti in enumerate(t)]
    return np.array(err).T
