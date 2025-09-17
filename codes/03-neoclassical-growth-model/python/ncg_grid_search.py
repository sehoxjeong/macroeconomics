""" 
SOLVING DETERMINISTIC NEOCLASSICAL GROWTH IN DISCRETE TIME WITH GRID SEARCH

AUTHOR: Seho Jeong, Sogang University
DATE: 2025-09-14
REFERENCES:
- Edmond, Chris. 2019. "Lectures on Macroeconomics (PhD Core)." University of Melbourne.
- Hong, Jay H. 2025. "Lectures on Topics in Macroeconomics." Seoul National University.
- Kopecky, Karen A. 2006. "Value Function Iteration." Computational Methods for Macroeconomics.
- Sargent, Thomax J., and John Stachurski. n.d. "Optimal Savings I: Value Function Iteration." Quantitative Economics with Python using JAX.
- Sargent, Thomas J., and John Stachurski. n.d. "Optimal Savings II: Alternative Algorithms." Quantitative Economics with Python using JAX.
"""

# Import necessary libraries.
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import while_loop, fori_loop

import matplotlib.pyplot as plt
from matplotlib import colors

from time import time
from functools import partial

# Gain extra precision by using 64-bit floats.
jax.config.update('jax_enable_x64', True)

# Define utility and technology.
# u = lambda c: jnp.log(c)        # utility function
# f = lambda k: A * (k ** alpha)  # production function

@jax.jit
def u(c):
    """ 
    Log utility function
    """
    return jnp.log(c)

@jax.jit
def f(k, params):
    """ 
    Cobb-Douglas production function
    """
    _, alpha, _, A = params
    return A * (k ** alpha)

def create_model():
    # Set model parameters.
    beta  = 0.96 # discount factor
    alpha = 1/3  # capital share in production function
    delta = 1    # capital depreciation rate
    A     = 1    # productivity

    # Set grid parameters.
    knum = 30    # number of points in grid for capital
    kmin = 1e-2  # min. capital
    kmax = 4     # max. capital

    kgrid = jnp.linspace(kmin, kmax, knum)

    params = (beta, alpha, delta, A)
    sizes  = (knum, )
    arrays = (kgrid, ) 

    C = (jnp.log(A) + (1 - alpha * beta) * jnp.log(1 - alpha * beta) + alpha * beta * jnp.log(alpha * beta)) / ((1 - beta) * (1 - alpha * beta))
    D = alpha / (1 - alpha * beta)
    g_true = lambda k: alpha * beta * A * (k ** alpha)
    v_true = lambda k: C + D * jnp.log(k)

    # Sanity check for the grid
    assert (f(kmin, params) > kmin) & (f(kmax, params) < kmax), 'Grid for capital is not valid.'

    return (params, sizes, arrays), (v_true, g_true)


@partial(jax.jit, static_argnums=(2, ))
def compute_reward(V, params, sizes, arrays):
    # Unpack parameters. 
    beta, _, delta, _ = params
    knum, = sizes
    kgrid, = arrays

    # Compute current rewards.
    k  = jnp.reshape(kgrid, (knum, 1)) # k[i]   -> k[i, ip]
    kp = jnp.reshape(kgrid, (1, knum)) # k'[ip] -> k'[i, ip]

    c = f(k, params) + (1 - delta) * k - kp

    # Calculate continuation rewards at all combinations of (k, kp).
    V = jnp.reshape(V, (1, knum))

    # Compute the right-hand side of the Bellman equation.
    RHS = jnp.where(c > 0, u(c) + beta * V, -jnp.inf)

    return RHS


@partial(jax.jit, static_argnums=(2, ))
def T(V, params, sizes, arrays):
    """  
    A Bellman operator
    """
    # Maximize over k' and get TV.
    return jnp.max(compute_reward(V, params, sizes, arrays), axis=1)


@partial(jax.jit, static_argnums=(2, ))
def get_policy(V, params, sizes, arrays):
    """ 
    Computes a policy function, returned as a set of indicies.
    """
    return jnp.argmax(compute_reward(V, params, sizes, arrays), axis=1)


@partial(jax.jit, static_argnums=(0, 2, 3))
def value_function_iteration(T,              # operator (callable)
                             v_0,            # initial condition
                             tol=1e-7,       # error tolerance
                             max_iter=1000): # max. iteration bound
    
    def body_func(iter_v_err):
        iteration, v, _ = iter_v_err
        v_new = T(v)
        error = jnp.max(jnp.abs(v_new - v))

        # Print interim results.
        print_cond = jnp.remainder(iteration, 10) == 0

        def true_func(args):
            iteration, error = args
            jax.debug.print('Iteration: {}, with error = {}', iteration, error)
            return ()
        
        def false_func(args):
            return ()
        
        jax.lax.cond(print_cond, true_func, false_func, (iteration, error))

        return iteration + 1, v_new, error
    
    def cond_func(iter_v_err):
        iteration, _, error = iter_v_err
        return jnp.logical_and(error > tol, iteration < max_iter)
    
    iteration, v, error = while_loop(cond_func, body_func, (1, v_0, np.inf))
    jax.debug.print('Iteration: {i}, with error = {e}', i=iteration, e=error)
    
    return v


@partial(jax.jit, static_argnums=(2, 3))
def policy_evaluation(V, g, params, sizes, arrays):
    beta, _, delta, _ = params
    knum, = sizes
    kgrid, = arrays

    # Get capital for next period from the policy function.
    kp = kgrid[g]

    # Calculate consumption implied by the policy.
    c = f(kgrid, params) + (1 - delta) * kgrid - kp

    return jnp.where(c > 0, u(c) + beta * V[g], -jnp.inf)


@partial(jax.jit, static_argnums=(0, 2, 3, 5))
def howard_policy_iteration(T, v_0, params, sizes, arrays, nH=5, tol=1e-8, max_iter=1000):

    def body_func(iter_v_err):
        iteration, v, _ = iter_v_err
        
        # Find the best policy for the current V.
        g = get_policy(v, params, sizes, arrays)

        # Howard improvement: Iterate the value function nH times.
        howard_body = lambda _, v: policy_evaluation(v, g, params, sizes, arrays)
        v_new = fori_loop(0, nH, howard_body, v)

        error = jnp.max(jnp.abs(v_new - v))

        return iteration + 1, v_new, error
    
    def cond_func(iter_v_err):
        iteration, _, error = iter_v_err
        return jnp.logical_and(error > tol, iteration < max_iter)
    
    # Run the while loop.
    iteration, v_star, error = while_loop(cond_func, body_func, (0, v_0, jnp.inf))
    jax.debug.print('HPI converged after {i} iterations with error {e}', i=iteration, e=error)

    return v_star


def solve_model(model, tol=1e-7, method='vfi'):

    # Unpack the model.
    params, sizes, arrays = model

    # Initialize guesses.
    V = jnp.zeros(sizes)
    g = jnp.zeros(sizes)

    T_temp = lambda v: T(v, params, sizes, arrays)

    if method == 'vfi':
        v_star = value_function_iteration(T_temp, V, tol=tol)
    elif method == 'hpi':
        v_star = howard_policy_iteration(T_temp, V, params, sizes, arrays)

    return v_star


### MAIN ###

model, solution = create_model()

# Unpack the model.
params, sizes, arrays = model
beta, alpha, delta, A = params
knum, = sizes
kgrid, = arrays
v_true, g_true = solution

# Solve using standard value function iteration.
print('Solving with standard Value Function Iteration (VFI) ...')
start_vfi = time()
v_star_vfi = solve_model(model, method='vfi')
g_star_vfi = get_policy(v_star_vfi, params, sizes, arrays)
end_vfi = time()
print(f'VFI completed in {end_vfi - start_vfi:.5e} seconds.')

# Solve using Howard improvement algorithm.
print('Solving with Howard Policy Iteration (HPI) ...')
start_hpi = time()
v_star_hpi = solve_model(model, method='hpi')
g_star_hpi = get_policy(v_star_hpi, params, sizes, arrays)
end_hpi = time()
print(f'VFI completed in {end_hpi - start_hpi:.5e} seconds.')

# Plot results.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

# LEFT PANEL
ax1.plot(kgrid, v_true(kgrid), color='k', label='True value')
ax1.plot(kgrid, v_star_vfi, color='dodgerblue', marker='x', label='Approx. by VFI')
ax1.plot(kgrid, v_star_hpi, color='crimson', marker='.', label='Approx. by HPI')

ax1.grid(color='gray', ls=':')
ax1.legend(fancybox=False, edgecolor='k', loc='lower right')
ax1.set_facecolor('#F5F5F5')
ax1.set_xlabel('$k$')
ax1.set_ylabel('$V(k)$')

# RIGHT PANEL
ax2.plot(kgrid, g_true(kgrid), color='k', label='True policy')
ax2.plot(kgrid, kgrid[g_star_vfi], color='dodgerblue', marker='x', label='Approx. by VFI')
ax2.plot(kgrid, kgrid[g_star_hpi], color='crimson', marker='.', label='Approx. by HPI')

ax2.grid(color='gray', ls=':')
ax2.legend(fancybox=False, edgecolor='k', loc='lower right')
ax2.set_facecolor('#F5F5F5')
ax2.set_xlabel('$k$')
ax2.set_ylabel('$g(k)$')

# fig, ax = plt.subplots()

# ax.plot(kgrid, (g_true(kgrid) - kgrid[g_star_vfi]) / g_true(kgrid))


plt.show()