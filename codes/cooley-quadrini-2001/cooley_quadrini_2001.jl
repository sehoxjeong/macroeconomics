# JULIA CODE REPLICATING RESULTS OF COOLEY & QUADRINI (2001, AER)

# Code by Seho Jeong, Sogang University
# January, 2025

# Model type A refers to II.C of the paper and type B refers to II.D of the paper.

##### I. LOAD PACKAGES.

##### II. PARAMETERIZATION

# Optimization parameters
model_type = "IID"
tol = 1e-6               # tolerance
maxiter = 100            # max. num. of value function integration

# Model parameters
β = 0.956                # intertemporal discount rate
r = 0.04                 # lending rate; risk-free
δ = 0.07                 # depriciation rate
ν = 0.975                # returns to scale
σ = 0.28                 # std. dev. of the shock ε
emin = 0                 # equity of the smallest firm
emax = 100               # equity of the largest firm
enum = 1000              # num. of grid points
lmin = 0                 # labor of the smallest firm
lmax = 2000              # labor of the largest firm
ξ = 0.01*emax            # default cost; X% of the equity value of the largest firm
λ = 0.3                  # equity issuance cost
# φ =                      # capital income share
# ζ = 0.1                  # fixed cost of production

if model_type == "IID"
    z0 = 0
    z1 = 0.428               # non-absorbing productivity in the case of iid shocks
    Γ = [1 0; 0.045 1-0.045] # transition matrix of z
elseif model_type == "PER"
    z0 = 0
    # z1 = # low productivity
    # z2 = # high productivity
    Γ = [1 0 0; 0.045 0.95 0.005; 0.045 0.005 0.045] # (?)
else
    println("Invalid model type. Please check if the `model_type` parameter is correctly specified.")
end

function print_model_parameters()
    println("========== MODEL PARAMETERS ==========")
end