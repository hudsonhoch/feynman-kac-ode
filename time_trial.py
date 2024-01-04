import time
import numpy as np
from solvers import DiffusionODESolver, ShootingMethodSolver

def run_trial(ndims, npoints=25):
    T = 2
    yT = np.ones(ndims)
    g = lambda t: -np.identity(ndims)
    h = lambda t: np.zeros((ndims, 1))
    A = lambda t: np.identity(ndims)
    D = lambda t: np.ones((ndims, 1)) * np.exp(-t)
    phi = lambda a,b: np.exp(b-a) * np.identity(ndims)
    solvers = {
        'Diffusion': DiffusionODESolver(A=A, D=D, T=T, yT=yT, phi=phi),
        'Shooting': ShootingMethodSolver(g, h, T=T, yT=yT),
    }
    solutions = {}
    for name,solver in solvers.items():
        start = time.time()
        t,y = solver.solve(npoints)
        duration = time.time()-start
        solutions[name] = {'t':t, 'y':y, 'duration':duration, 'solver':solver}

    return solutions



ntrials = 100
for i in range(10):

    print(f'ndims = {2**i}')
    solutions = run_trial(2**i)
    durations = {name: [sol['duration']] for name,sol in solutions.items()}
    for _ in range(ntrials-1):
        solutions = run_trial(2**i)
        for name,sol in solutions.items():
            durations[name].append(sol['duration'])

    for name,durs in durations.items():
        print(f'  {name}: {np.mean(durs):.4f} seconds')
    print()