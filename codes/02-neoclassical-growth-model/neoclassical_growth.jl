# Neoclassical Growth Model 
# Wriiten by Seho Jeong, Sogang University
# September 2025

println("Initiating the Program - Neoclassical Growth Model")

# Load essential packages.
using Plots
using ProgressBars

# Numerical parameters
max_iter = 500   # maximum number of iterations
tol      = 1e-7  # VFI tolerance
penalty  = 10^16 # for penalizing constraint violations

# Model parameters
β = 0.95 # discount factor
δ = 1    # capital depriciation rate
A = 1    # productivity
α = 1/3  # returns-to-scale parameter

# Utility and technology
utility(c) = log(c)
production(k) = A * k^α

# Analytic solution
D = α / (1 - α * β)
C = (log(A) + (1 - α*β)*log(1 - α*β) + α*β*log(α*β)) / ((1 - β) * (1 - α*β))
Vtrue(k) = C + D * log(k)
gtrue(k) = (β*D / (1 + β*D)) * A * k^α

# Set grids.
knum = 100             # number of nodes for k grid
kmin = tol             # effectively zero
kmax = 40              # effective upper bound on k grid
kgrid = collect(range(kmin, kmax, knum)) # linearly spaced

# Return function
c = zeros(knum, knum)
for j in 1:knum
    c[:, j] = production.(kgrid) + (1 - δ) * kgrid .- kgrid[j]
end

# Penalize violations of feasibility constraints.
c = c .* (c .>= 0)
u = utility.(c) - penalty * (c .< 0) .+ 1e-10

# Bellman operator
function T(V)
    # RHS of Bellman equation
    RHS = u .+ β * kron(ones(knum, 1), V')

    # Maximize over RHS to get TV.
    maximum(V, )
end

println("Initiating the loop")
V = zeros(length(kgrid))


println("The program ends - Neoclassical Growth Model")