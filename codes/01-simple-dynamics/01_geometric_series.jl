# Load the essential packages.
using LinearAlgebra, Statistics
using Distributions, LaTeXStrings, Plots, Random, Symbolics, Latexify

println("Initiated the program: Geometric Series")

# True present value of a finite lease
function finite_lease_pv_true(T, g, r, x_0)
    G = 1 + g 
    R = 1 + r 
    return (x_0 * (1 - G^(T + 1) * R^(-T - 1))) / (1 - G * R^(-1))
end

# First approximation for our finite lease.
function finite_lease_pv_approx(T, g, r, x_0)
    x_0 * (T + 1) + x_0 * r * g * (T + 1) / (r - g)
end

# Second approximation for our finite lease.
finite_lease_pv_approx_2(T, g, r, x_0) = (x_0 * (T + 1))




println("The program ends: Geometric Series")