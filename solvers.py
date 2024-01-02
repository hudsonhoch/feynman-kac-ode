from typing import Callable

import numpy as np
import scipy.stats as stats
import scipy.linalg as slinalg
from scipy.integrate import quad, quad_vec
from scipy.optimize import minimize


class DiffusionODESolver:

    def __init__(
            self,
            T: float,
            yT: np.ndarray,
            A: Callable[[float], np.ndarray],
            D: Callable[[float], np.ndarray],
            V: Callable[[float], np.ndarray] = None,
            phi: Callable[[float,float], np.ndarray] = None
        ):
        A0 = A(0)
        D0 = D(0)

        if V is None:
            self.V = lambda t: np.zeros(A0.shape)
        else:
            self.V = V
    
        V0 = self.V(0)
        assert D0.shape[1] == 1
        assert A0.shape[0] == D0.shape[0]
        assert A0.shape[1] == D0.shape[0]
        assert V0.shape[0] == D0.shape[0]
        assert V0.shape[1] == D0.shape[0]

        self.A = A
        self.D = D
        self.T = T  # end point
        self.yT = yT if yT.ndim == 2 else yT[:,None]  # terminal condition
        self.N = D0.shape[0]  # size of system

        if phi is None:
            self.phi = lambda a,b: np.linalg.expm(quad_vec(A, a, b)[0])
        else:
            self.phi = phi


    def mu(self, t: float, y: np.ndarray, s: float) -> np.ndarray:
        term1 = self.phi(s,t) @ y
        integrand = lambda u: self.phi(u,t) @ self.D(u)
        term2 = quad_vec(integrand, s, t)[0]
        return term1 + term2


    def var(self, t: float, y: np.ndarray, s: float):
        def integrand(u):
            p = self.phi(u,t)
            return p.T @ p
        return quad_vec(integrand, s, t)[0]
    

    def transition(self, x: np.ndarray, t: float, y: np.ndarray, s: float) -> float:
        variance = self.var(t,y,s)
        mean = self.mu_(t,y,s)
        denom = np.sqrt(2*np.pi*variance)
        exp = np.exp(-(x - mean)**2 / (2*variance))
        return exp/denom
    

    def simulate_diffusion(self, npaths: int, npoints: int, end_time: float = None):
        if end_time is None:
            end_time = self.T

        times = np.linspace(0, end_time, npoints)
        dt = times[1]
        distribution = stats.norm(scale=np.sqrt(dt))
        dW = lambda: distribution.rvs((self.N,npaths))

        Xt = np.zeros((self.N,npaths,npoints))
        for i,t in enumerate(times[:-1],1):
            x = Xt[:,:,i-1]
            Xt[:,:,i] = x + (self.A(t) @ x + self.D(t)) * dt + dW()

        return times, Xt
    

    def _deterministic_bridge_step(self, y: np.ndarray, t: float):
        term1 = self.A(t) @ y + self.D(t)
        term2 = (self.yT - self.mu(self.T,y,t))
        term2 = self.phi(t,self.T) @ term2
        term2 = np.linalg.inv(self.var(self.T,y,t)) @ term2
        return term1 + term2


    def simulate_bridge(self, npaths: int, npoints: int, end_time: float = None) -> (np.ndarray, np.ndarray):
        if end_time is None:
            end_time = self.T

        times = np.linspace(0, end_time, npoints)
        dt = times[1]
        distribution = stats.norm(scale=np.sqrt(dt))
        dW = lambda: distribution.rvs((self.N,npaths))

        Yt = np.zeros((self.N,npaths,npoints))
        for i,t in enumerate(times[:-1],1):
            y = Yt[:,:,i-1]
            Yt[:,:,i] = y + self._deterministic_bridge_step(y,t) * dt + dW()

        return times, Yt
    

    def solve(self, npoints: int) -> (np.ndarray, np.ndarray):
        times = np.linspace(0, self.T, npoints)
        dt = times[1]

        Yt = np.zeros((self.N,1,npoints))
        weight = np.empty((npoints-1, self.N, self.N))
        for i,t in enumerate(times[:-1],1):
            y = Yt[:,:,i-1]
            weight[i-1] = slinalg.expm(self.V(t))
            Yt[:,:,i] = y + self._deterministic_bridge_step(y,t) * dt

        Yt[:,:,:-1] = np.einsum('kij,jmk->imk', weight, Yt[:,:,:-1])
        Yt[:,0,[-1]] = self.yT
        return times, Yt[:,0,:] 




class OMPolySolver:

    def __init__(
            self,
            A: Callable[[float], np.ndarray],
            D: Callable[[float], np.ndarray],
            T: float,
            yT: np.ndarray,
            n: int,
            method = 'Newton-CG'
        ):
        A0 = A(0)
        D0 = D(0)
        assert A0.shape[0] == D0.shape[0]
        assert A0.shape[1] == D0.shape[0]
        assert D0.shape[1] == 1

        self.A = A
        self.D = D
        self.T = T  # end point
        self.yT = yT if yT.ndim == 2 else yT[:,None]  # terminal condition
        self.n = n  # degree of polynomial approximation
        self.N = D0.shape[0]  # size of system
        self.K = np.zeros((self.N, self.n-1))  # initial polynomial coefficients
        self.K[:,[-1]] = self.yT / T  # initial guess is line between z(0) and z(T)

        self.method = method
        self.solved = False

        x = np.repeat(np.arange(1,n).reshape((-1,1)), n-1, axis=1)
        x = np.maximum(0, x - x.T + 1)
        x = np.hstack((np.ones((self.n-1,1)), x, np.zeros((self.n-1,1))))
        self._coefs = x.cumprod(axis=1)
        self._pwrs = x[:,1:]
        self.Eta = T ** self._pwrs[:,[0]]


    def __call__(self, t) -> np.ndarray:
        inner = self.yT - self.K @ self.Eta
        return np.outer(inner, t**self.n / self.T**self.n) + self.K @ self.eta(t)
    

    def eta(self, t, order: int = 0) -> np.ndarray:
        assert order >= 0
        return self._coefs[:,[min(self.n, order)]] * t**(self._pwrs[:,[min(self.n-1, order)]])
    

    def deriv(self, t, order: int = 1) -> np.ndarray:
        assert order >= 1
        inner = self.yT - self.K @ self.Eta
        coef = np.prod([self.n-i for i in range(order)])
        return np.outer(inner, coef * t**(self.n-order) / self.T**self.n) + self.K @ self.eta(t, order)


    def F(self, t: float, K: np.ndarray) -> float:
        inner = self.yT - K @ self.Eta
        eta = t ** np.arange(self.n)
        z_hat = (t**self.n / self.T**self.n) * inner + K @ self.eta(t)
        z_hat_dot = (self.n * t**(self.n-1) / self.T**self.n) * inner + K @ self.eta(t, 1)
        return z_hat_dot - self.A(t) @ z_hat - self.D(t)


    def _objective(self, K: np.ndarray, jac=False) -> float:
        ofv = quad(lambda t: np.linalg.norm(self.F(t, K)), 0, self.T)[0]
        if jac:
            return ofv, self._jacobian(K).flatten()
        else:
            return ofv 
    

    def _normF_gradient(self, t: float, K: np.ndarray) -> np.ndarray:
        Ft = self.F(t,K)
        term1 = Ft @ (self.eta(t, 1).T - self.n * t**(self.n-1) / self.T**(self.n) * self.Eta.T)
        term2 = self.A(t) @ Ft @ (self.eta(t).T - t**(self.n) / self.T**(self.n) * self.Eta.T)
        return 2 * (term1 - term2)
    

    def _jacobian(self, K: np.ndarray) -> np.ndarray:
        return quad_vec(self._normF_gradient, 0, self.T, args=(K,))[0]
    

    def solve(self) -> None:
        f = lambda K: self._objective(K.reshape((self.N,self.n-1)), jac=True)
        self.result = minimize(f, self.K.flatten(), jac=True, method=self.method)
        self.K = self.result.x.reshape((self.N, self.n-1))
        self.solved = True