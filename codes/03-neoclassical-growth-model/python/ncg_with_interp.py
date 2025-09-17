""" 
SOLVING DETERMINISTIC NEOCLASSICAL GROWTH IN DISCRETE TIME WITH INTERPOLATION

AUTHOR: Seho Jeong, Sogang University
DATE: 2025-09-14
REFERENCES:
- Hong, Jay H. 2025. "Lectures on Topics in Macroeconomics." Seoul National University.
- Schumaker, Larry L. 1983. "On Shape Preserving Quadratic Spline Interpolation" SIAM Journal of Numerical Analysis 20(4): 854-864.
"""

# Import libraries.
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator, CubicSpline
from scipy.optimize import minimize_scalar
from schumaker import SchumakerSpline1983

import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Callable, Tuple
from time import time

# Some useful functions
def running_time(func):

    def wrapper(*args, **kwargs):
        
        start = time()
        out = func(*args, **kwargs)
        end = time()

        print(f'Total running time = {end - start} seconds')

        return out
    
    return wrapper


# Model primitives
@dataclass
class Parameters:
    beta: float  = 0.96 # time discount factor
    alpha: float = 1/3  # capital share in production function
    delta: float = 1.0  # capital depreciation rate
    A: float     = 1.0  # productivity

@dataclass
class Grid:

    knum: int   = 30
    kmin: float = 1e-2
    kmax: float = 4.0

    def build(self) -> np.ndarray:
        return np.linspace(self.kmin, self.kmax, self.knum)

def u(c: float) -> float:
    """ 
    Log utility
    """
    return np.log(c)

def f(k: float, params: Parameters) -> float:
    """
    Cobb-Douglas production
    """
    return params.A * (k ** params.alpha)

# Define interpolator.
def interpolator(kind: str, k: np.ndarray, V: np.ndarray) -> Callable[[float], float]:

    if kind == 'pchip':
        vf = PchipInterpolator(k, V, extrapolate=False)

    elif kind == 'linear':
        vf = interp1d(k, V, kind=kind, bounds_error=True)

    elif kind == 'cubic':
        vf = CubicSpline(k, V, bc_type='not-a-knot')

    elif kind == 'schumaker':
        vf = SchumakerSpline1983(k, V)
    
    else:
        raise ValueError('Unknown interpolator kind.')
    
    return vf

def T(V: np.ndarray,
      params: Parameters,
      kgrid: np.ndarray,
      interpolator_kind: str = 'linear',
      tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    
    beta, delta = params.beta, params.delta
    kmin, kmax = kgrid[0], kgrid[-1]

    # Build the interpolator.
    vf = interpolator(interpolator_kind, kgrid, V)

    # Precompute income and feasible bounds for k'.
    y = f(kgrid, params) + (1 - delta) * kgrid
    ub = np.minimum(kmax, y-tol) # upper bound for feasible k'

    TV = np.empty_like(V)
    g = np.empty_like(V)

    for i, k in enumerate(kgrid):
        if ub[i] <= kmin:
            TV = -np.inf
            g[i] = kmin
            continue

        def neg_reward(kp: float) -> float:
            c = y[i] - kp
            if c <= 0:
                return np.inf
            return - (u(c) + beta * vf(kp))
        
        result = minimize_scalar(neg_reward, bounds=(kmin, ub[i]), method='bounded')
        TV[i] = -result.fun
        g[i] = result.x

    return TV, g

def howard_evaluate(V: np.ndarray,
                    g: np.ndarray,
                    params: Parameters,
                    kgrid: np.ndarray,
                    interpolator_kind: str,
                    nH: int) -> np.ndarray:
    # Standard value function iteration if nH == 0.
    if nH == 0:
        return V
    
    beta, delta = params.beta, params.delta
    y = f(kgrid, params) + (1 - delta) * kgrid
    c = np.maximum(y - g, 1e-12)
    utility = np.log(c)

    Vh = V.copy()
    for _ in range(nH):
        vf = interpolator(interpolator_kind, kgrid, Vh)
        Vh = utility + beta * vf(g)
    
    return Vh

@running_time
def solve_model(params: Parameters,
                grid: Grid,
                interpolator_kind: str = 'linear',
                tol: float = 1e-7,
                max_iter: int = 1000,
                howard_iter: int = 25,
                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    kgrid = grid.build()

    # Sanity check
    assert (f(grid.kmin, params) > grid.kmin) and (f(grid.kmax, params) < grid.kmax), 'Entered capital grid is not feasible.'

    V = np.zeros_like(kgrid) # initial guess

    for iteration in range(max_iter):
        
        # Standard VFI step
        TV, g = T(V, params, kgrid, interpolator_kind)

        # Howard improvement step
        Vh = howard_evaluate(TV, g, params, kgrid, interpolator_kind, nH=howard_iter)

        error = np.max(np.abs(Vh - V))
        if verbose and (iteration % 10 == 0 or error < tol):
            print(f'Iteration {iteration:4d}: error = {error:.3e}')
        
        # Update the value.
        V[:] = Vh

        # Convergence check
        if error < tol:
            print(f'Converged in iteration {iteration:4d}: error = {error:.3e}')
            break

    return kgrid, V, g


if __name__ == '__main__':

    params = Parameters(beta=0.96, alpha=1/3, delta=1.0, A=1.0)
    kgrid = Grid(knum=30, kmin=1e-2, kmax=4.0)

    # Closed-form for δ = 1
    C = (np.log(params.A) + (1 - params.alpha * params.beta) * np.log(1 - params.alpha * params.beta) + params.alpha * params.beta * np.log(params.alpha * params.beta)) / ((1 - params.beta)*(1 - params.alpha * params.beta))
    D = params.alpha / (1 - params.alpha * params.beta)
    v_true = lambda k: C + D * np.log(k)
    g_true = lambda k: params.alpha * params.beta * params.A * (k ** params.alpha)

    k, V1, g1 = solve_model(params, kgrid, interpolator_kind='linear', tol=1e-8, max_iter=1000, howard_iter=0, verbose=True)
    _, V2, g2 = solve_model(params, kgrid, interpolator_kind='schumaker', tol=1e-8, max_iter=1000, howard_iter=0, verbose=True)
    _, V3, g3 = solve_model(params, kgrid, interpolator_kind='cubic', tol=1e-8, max_iter=1000, howard_iter=0, verbose=True)

    k, V4, g4 = solve_model(params, kgrid, interpolator_kind='linear', tol=1e-8, max_iter=1000, howard_iter=50, verbose=True)
    _, V5, g5 = solve_model(params, kgrid, interpolator_kind='schumaker', tol=1e-8, max_iter=1000, howard_iter=50, verbose=True)
    _, V6, g6 = solve_model(params, kgrid, interpolator_kind='cubic', tol=1e-8, max_iter=1000, howard_iter=50, verbose=True)
    
    # Plot value functions and policy functions.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(k, v_true(k), 'k-', label='Closed-form V')
    ax1.plot(k, V1, marker='.', color='red', label='Linear', alpha=0.7)
    ax1.plot(k, V2, marker='.', color='green', label='Schumaker', alpha=0.7)
    ax1.plot(k, V3, marker='.', color='blue', label='Cubic', alpha=0.7)

    ax1.legend(fancybox=False, edgecolor='k', loc='lower right')
    ax1.set_facecolor('#F5F5F5')
    ax1.set_xlabel('$k$')
    ax1.set_ylabel('$V(k)$')

    ax2.plot(k, g_true(k), 'k-', label='Closed-form V')
    ax2.plot(k, g1, marker='.', color='red', label='Linear', alpha=0.7)
    ax2.plot(k, g2, marker='.', color='green', label='Schumaker', alpha=0.7)
    ax2.plot(k, g3, marker='.', color='blue', label='Cubic (Not-a-knot)', alpha=0.7)

    ax2.legend(fancybox=False, edgecolor='k', loc='lower right')
    ax2.set_facecolor('#F5F5F5')
    ax2.set_xlabel('$k$')
    ax2.set_ylabel('$g(k)$')

    # Plot approximation error of policy function.
    fig, ax = plt.subplots()

    ax.axhline(0, color='k', ls=':')
    ax.plot(k, (g1 - g_true(k)) / g_true(k) * 100, marker='.', color='red', alpha=0.7, label='Linear')
    ax.plot(k, (g2 - g_true(k)) / g_true(k) * 100, marker='.', color='green', alpha=0.7, label='Schumaker')
    ax.plot(k, (g3 - g_true(k)) / g_true(k) * 100, marker='.', color='blue', alpha=0.7, label='Cubic (Not-a-knot)')

    ax.legend(fancybox=False, edgecolor='k', loc='lower right')
    ax.set_facecolor('#F5F5F5')
    ax.set_xlabel('$k$')
    ax.set_ylabel('Approximation error (%)')
    ax.set_xmargin(0)

    plt.show()