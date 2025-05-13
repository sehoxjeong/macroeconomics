import numpy as np
from scipy.stats import norm

def tauchen(n, rho, sigma, mu=0, m=3):
    """ 
    Approximate an AR(1) process by a finite Markov chain using Tauchen's method.
    """
    xgrid = np.zeros(n)
    xprob = np.zeros((n, n))

    a = (1 - rho) * mu 

    xgrid[-1] = m * np.sqrt(sigma ** 2 / (1 - (rho ** 2)))
    xgrid[0] = -1 * xgrid[-1]
    xstep = (xgrid[-1] - xgrid[0]) / (n - 1)

    for i in range(1, n):
        xgrid[i] = xgrid[0] + xstep * i

    xgrid = xgrid + a / (1 - rho)

    for j in range(n):
        for k in range(n):
            
            if k == 0:
                xprob[j, k] = norm.cdf( (xgrid[0] - a - rho * xgrid[j] + xstep / 2) / sigma )
            elif k == n-1:
                xprob[j, k] = 1 - norm.cdf( (xgrid[-1] - a - rho * xgrid[j] - xstep / 2) / sigma )
            else:
                up = norm.cdf((xgrid[k] - a - rho * xgrid[j] + xstep / 2) / sigma)
                down = norm.cdf((xgrid[k] - a - rho * xgrid[j] - xstep / 2) / sigma)
                xprob[j, k] = up - down
    
    return xgrid, xprob