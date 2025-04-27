import numpy as np
from scipy.stats import norm

def tauchen_hussey_log(n, rho, sigma, mu=0, floden=False):
    """
    Implements the Tauchen-Hussey (1991) method to discretize a lognormal AR(1) process

        log(y') = μ + ρ log(y) + σϵ,    ϵ ~ N(0, 1).

    Parameters:
    - n: number of nodes
    - rho: persistence of the process
    - sigma: innovation std. dev.
    - mu: unconditional mean of log y
    - floden: whether to apply Floden's correction
    
    Returns:
    - nodes
    - weights
    - P
    """
    # Step 1: Floden correction
    if floden:
        w = 0.5 + rho / 4
        sigx = sigma / np.sqrt(1 - rho**2) # unconditional standard deviation
        floden_sigma = w * sigma + (1 - w) * sigx
    else:
        floden_sigma = sigma

    # Step 2: Lookup quadrature nodes and weights.
    nodes, weights = np.polynomial.hermite.hermgauss(n)

    # Step 3: Convert to lognormal shock values
    y_vals = np.exp(mu + np.sqrt(2) * floden_sigma * nodes)

    # Step 3: Construct transition probability matrix.
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mean = rho * np.log(y_vals[i]) + (1 - rho) * mu # conditional mean in logs
            P[i, j] = norm.pdf(np.log(y_vals[j]), mean, sigma) * weights[j] / norm.pdf(nodes[j], 0, 1)

    # Step 4: Normalize rows to ensure probabilities sum to 1.
    P /= P.sum(axis=1, keepdims=True)

    return y_vals, P