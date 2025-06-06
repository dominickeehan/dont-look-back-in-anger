using Random, Distributions, Statistics, StatsBase
using JuMP, Gurobi
using Plots
using ProgressBars

env = Gurobi.Env()
GRBsetintparam(env, "OutputFlag", 0)
Linear_Optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

"""
Compute the first-order Wasserstein distance between two one-dimensional empirical distributions.
"""
function W1(P, Q)

    # P, Q := [points, weights] .

    m = length(P[1])
    n = length(Q[1])

    d = [abs(ξ - ζ) for ξ in P[1], ζ in Q[1]]

    model = Model(Linear_Optimizer)

    @variable(model, γ[1:m, 1:n] >= 0)

    @constraint(model, [i=1:m], sum(γ[i,j] for j in 1:n) == P[2][i])
    @constraint(model, [j=1:n], sum(γ[i,j] for i in 1:m) == Q[2][j])

    @objective(model, Min, sum(γ[i,j] * d[i,j] for i in 1:m, j in 1:n))

    optimize!(model)

    try; return objective_value(model); catch; return Inf; end

end


normalise(x) = x/sum(x)

support = LinRange(-50,50,1000)

P = [Vector(LinRange(-25,25,100)), normalise(Vector(LinRange(0.0,1.0,100)))]

ε = 5
number_of_distributions = 100000
number_of_points = 30

Qs = [[zeros(number_of_points), zeros(number_of_points)] for _ in 1:number_of_distributions]
distance_to_Ps = zeros(length(Qs))

Threads.@threads for i in ProgressBar(eachindex(Qs))
    n = rand(1:number_of_points)
    points = rand(support, n)
    weights = rand(Uniform(0,1), n)
    weights .= normalise(weights)

    Qs[i] = [points, weights]

    distance_to_Ps[i] = W1(P,Qs[i])

end



bins = 5
alpha = 0.0
fillalpha = 0.005
plt = plot(xlims=(support[1],support[end]))

for i in ProgressBar(eachindex(Qs))
    if distance_to_Ps[i] <= ε
        stephist!(Qs[i][1], weights=Qs[i][2], color=:red, bins=bins, alpha=alpha, fill=true, fillalpha=fillalpha, labels=nothing,)

    end

end

stephist!(P[1], weights=P[2], bins=bins, color=:black, alpha=1, fill=true, fillalpha=0.1, labels=nothing,)
display(plt)


