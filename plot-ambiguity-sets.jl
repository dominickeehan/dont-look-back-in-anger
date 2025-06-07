using Random, Distributions, Statistics, StatsBase
using JuMP, Gurobi
using Plots
using ProgressBars

Random.seed!(42)

env = Gurobi.Env()
GRBsetintparam(env, "OutputFlag", 0)
Linear_Optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

"""
Compute the second-order Wasserstein distance between two one-dimensional empirical distributions.
"""
function W2(P, Q)

    # P, Q := [points, weights] .

    m = length(P[1])
    n = length(Q[1])

    d = [abs(ξ - ζ)^2 for ξ in P[1], ζ in Q[1]]

    model = Model(Linear_Optimizer)

    @variable(model, γ[1:m, 1:n] >= 0)

    @constraint(model, [i=1:m], sum(γ[i,j] for j in 1:n) == P[2][i])
    @constraint(model, [j=1:n], sum(γ[i,j] for i in 1:m) == Q[2][j])

    @objective(model, Min, sum(γ[i,j] * d[i,j] for i in 1:m, j in 1:n))

    optimize!(model)

    try; return sqrt(objective_value(model)); catch; return Inf; end

end

include("weights.jl")

normalise(x) = x/sum(x)

support = LinRange(-1000,1000,1000)
number_of_points = 50
P = [Vector(LinRange(-50,50,number_of_points)), W2_concentration_weights(number_of_points, 0.01)]

ε = 20
number_of_distributions = 100000

Qs = [[zeros(number_of_points), zeros(number_of_points)] for _ in 1:number_of_distributions]
plot_Qs = zeros(length(Qs))

Threads.@threads for i in ProgressBar(eachindex(Qs))
    n = rand(1:number_of_points)
    points = rand(support, n)
    weights = rand(Uniform(0,1), n)
    weights .= normalise(weights)

    Qs[i] = [points, weights]

    plot_Qs[i] = ifelse(W2(P,Qs[i]) <= ε, 1, 0)

end



bins = 10
alpha = 0.0
fillalpha = 1/sum(plot_Qs)
plt = plot(xlims=(support[1],support[end]))

for i in ProgressBar(eachindex(Qs))
    if plot_Qs[i] == 1
        stephist!(Qs[i][1], weights=Qs[i][2], bins=bins, normalise=true, color=:red, alpha=alpha, fill=true, fillalpha=fillalpha, labels=nothing,)

    end

end

stephist!(P[1], weights=P[2], bins=bins, normalise=true, color=:black, alpha=1, fill=false, labels=nothing,)
display(plt)


