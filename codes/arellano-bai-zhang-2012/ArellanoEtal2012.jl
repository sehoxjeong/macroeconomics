"""
TITLE: Arellano et al. (2012) Replication
AUTHOR: Seho Jeong, Sogang University
DESCRIPTION:
- Replicate Arellano et al. (2012) with Julia.
REFERENCES:
- Arellano, Cristina, Yan Bai, and Jing Zhang. 2012. "Firm Dynamics and Financial Development." Journal of Monetary Economics, 59: 533-49.
"""

# Import packages
using LinearAlgebra, Statistics
using LaTeXStrings, QuantEcon, Plots, Random


function create_model(; β=0.96,   # discount factor
                        r=0.04,   # interest rate
                        δ=0.1,    # capital depreciation rate
                        α=0.65,   # technology; returns to scale
                        γ=0.3,    # equity issuance cost
                        ψ=0.25,   # capital loss after default
                        θ=0.072,  # death rate
                        ρ=0.86,   # shock persistence
                        c=0.55,   # permanent productivity
                        σ=0.525,  # stochastic shock variance
                        φ=0.001,  # capital adjustment cost
                        ξ=0.01,   # credit cost
                        K0=0.002, # entrant starting capital,
                        γe=0.13)  # entrant equity issuance cost
    # Create grids.

    # Define value functions.
end

y(z, k, model) = z * k^model.α # output function of an operating firm

# function one_step_update!(model, EV)

# end