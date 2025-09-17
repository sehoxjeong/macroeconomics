%{
SOLVING DETERMINISTIC NEOCLASSICAL GROWTH IN DISCRETE TIME WITH GRID SEARCH

AUTHOR: Seho Jeong, Sogang University
DATE: 2025-09-14
REFERENCES:
- Edmond, Chris. 2019. "Lectures on Macroeconomics (PhD Core)." University of Melbourne.
- Hong, Jay H. 2025. "Lectures on Topics in Macroeconomics." Seoul National University.
%}

clear all; clc

% Model parameters
alpha = 1/3;        % capital's share in production function
beta  = 0.95;       % time discount factor
delta = 0.05;       % depreciation rate
sigma = 1;          % CRRA (=1/IES)
rho   = (1/beta)-1; % implied rate of time preference

kstar = (alpha / (rho + delta))^(1/(1-alpha)); % steady state
kbar  = (1/delta)^(1/(1-alpha));

% Numerical parameters
max_iter = 500;   % maximum number of iterations
tol      = 1e-7;  % treat numbers smaller than this as zero
penalty  = 10^16; % for penalizing constraint violations

% Set up the grid of capital stocks.
knum = 1001; % number of nodes for k grid
kmin = tol;  % effectively zero
kmax = kbar; % effective upper bound on k grid

kgrid = linspace(kmin, kmax, knum); % linearly spaced

% Return function
c = zeros(knum, knum);
for j=1:knum,
    c(:, j) = (kgrid.^alpha) + (1 - delta)*kgrid - kgrid(j);    
end

% Penalize violations of feasibility constraints.
violations = (c <= 0);
c = c .* (c >= 0) + eps;

if sigma == 1,
    u = log(c) - penalty * violations;
else
    u = (1 / (1 - sigma)) * (c.^(1 - sigma) - 1) - penalty * violations;
end

% Solve Bellman equation by value function iteration.

% Initial guess
V = zeros(knum, 1);

hold on
% Iterate on Bellman operator
for iter=1:max_iter
    % RHS of Bellman equation
    RHS = u + beta * kron(ones(knum, 1), V');

    % Maximize over this to get Tv.
    [TV, argmax] = max(RHS, [], 2);

    % Policy that attains the maximum
    g = kgrid(argmax);

    % Check if converged.
    error = norm(TV - V, inf);
    if error < tol
        fprintf('%4i %6.2e \n', [iter, error])
        break
    end

    V = TV;
    if rem(iter, 10) == 0
        fprintf('%4i %6.2e \n', [iter, error]);
    end
end

plot(kgrid, V)
hold off


% plot(kgrid, g)