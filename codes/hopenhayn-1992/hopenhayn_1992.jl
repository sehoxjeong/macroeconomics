"""
Hopenhayn (1992) Industry Equilibrium Model

AUTHOR:     Seho Jeong, Sogang University
DATE:       March 27, 2025
REFERENCES:
- Hopenhayn, Hugo A. "Entry, Exit, and Firm Dynamics in Long Run Equilibrium." Econometrica, 60(5): 1127-1150.
"""

# I - Import packages.
using LinearAlgebra, Plots

# II - Model struct
mutable struct Hopenhayn1992Model

    # 1. Parameters
    β = 0.96        # discount factor
    c_f = 0.1       # fixed operating cost
    c_e = 0.1       # entry cost

    # 2. Grids
    φnum = 200
end