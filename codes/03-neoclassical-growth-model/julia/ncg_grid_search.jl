#=
SOLVING DETERMINISTIC NEOCLASSICAL GROWTH MODEL IN DISCRETE TIME USING GRID SEARCH

AUTHOR: Seho Jeong, Sogang University
DATE: 2025-09-14
REFERENCES:
- Edmond, Chris. 2019. "Lectures on Macroeconomics (PhD Core)." University of Melbourne.
- Hong, Jay H. 2025. "Lectures on Topics in Macroeconomics." Seoul National University.
=#

using LinearAlgebra
using Plots, LaTeXStrings
using BenchmarkTools

function NeoclassicalGrowth()
    # Set model parameters.
    β = 0.96 # time discount factor
    α = 0.4  # capital share in production function
    A = 1.0  # productivity
    δ = 1.0  # capital depreciation rate

    knum = 1001
    kmin = 1e-7
    kmax = 4.0

    kgrid = collect(range(kmin, kmax, knum))

    return (; β, α, A, δ, knum, kgrid)
end

# Solve Bellman equation by value function iteration.
function value_function_iteration(params; max_iter=1000, tol=1e-7, howard=false)

    # Unpack parameters.
    β, α, A, δ, knum, kgrid = params

    # Define utility and technology.
    u(c) = log(c)
    f(k) = A * k^α

    # Compute consumption matrix.
    c = f.(kgrid) .- (1 - δ) * kgrid .- kgrid'

    plt = heatmap(c, xlabel=L"k'", ylabel=L"k") # plot the matrix to check if it is computed appropriately.

    # Penalize violations of feasibility constraints.
    violations = c .<= 0
    c = c .* (c .>= 0) .+ 1e-10
    utility = u.(c) .- 1e16 * violations

    # Guess V.
    V = zeros(knum) # initial guess for value function
    g = zeros(knum)

    # Iterate on bellman operator.
    for iter in 1:max_iter
        
        # RHS of Bellman equation
        RHS = utility .+ β * V

        # Maximize over the RHS to get TV and the policy g.
        TV = maximum(RHS, dims=2)
        amax = argmax(RHS, dims=2)
        g = kgrid[amax]            # policy that attains the maximum

        # Check if converged.
        error = maximum(abs.(TV .- V))

        if mod(iter, 10) == 0
            println("Iteration $iter with error $error")
        end

        if error < tol
            println("Converged : iteration = $iter; error = $error")
            break
        end

        V = copy(TV)
    end

    return V, g
end

params = NeoclassicalGrowth()
@time V, g = value_function_iteration(params)

# Plot the results
result1 = plot(kgrid, V, xlabel=L"k", ylabel=L"V(k)")