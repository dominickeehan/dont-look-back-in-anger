using Random, Distributions, Statistics, StatsBase
using JuMP, Gurobi
using Plots
using ProgressBars

Random.seed!(42)

env = Gurobi.Env()
GRBsetintparam(env, "OutputFlag", 0)
Linear_Optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

function in_W2_ball(P, ε, Q)

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

    try; return ifelse(sqrt(objective_value(model)) <= ε, 1, 0); catch; return 0; end

end

function in_W2_intersection(points, ε, ϱ, Q)

    # Q := [points, weights] .

    K = length(points)
    n = length(Q[1])

    for k in 1:K
        if sqrt(sum(Q[2][j] * abs(points[k] - Q[1][j])^2 for j in 1:n)) > ε + (K-k+1)*ϱ
            return 0 
        
        end
    end

    return 1

end

include("weights.jl")

normalise(x) = x/sum(x)

function mean_and_std(Q)

    return sum(Q[2][i]*Q[1][i] for i in 1:length(Q[1])), std(Q[1], Weights(Q[2]))

end

support = [-30,30]
number_of_points = 10

number_of_distributions = 10000

plt = plot(xlims=(support[1],support[end]), ylims=(0,30))

for ε in [20, 15, 10, 5]

    ϱ = ε/10

    P = [Vector(LinRange(-5,5,10)), W2_concentration_weights(10, ϱ/ε)]

    Qs = [[zeros(number_of_points), zeros(number_of_points)] for _ in 1:number_of_distributions]
    Threads.@threads for i in ProgressBar(eachindex(Qs))
        local n = number_of_points
        local μ = rand(Uniform(support[1],support[end]))
        local σ = rand(Uniform(0,30))
        local points = rand(Normal(μ, σ), n)

        Qs[i] = [points, 1/n*ones(n)]

    end

    plot_ball_Qs = zeros(length(Qs))
    Threads.@threads for i in ProgressBar(eachindex(Qs))
        plot_ball_Qs[i] = in_W2_ball(P,ε,Qs[i])

    end

    plot_intersection_Qs = zeros(length(Qs))
    Threads.@threads for i in ProgressBar(eachindex(Qs))
        plot_intersection_Qs[i] = in_W2_intersection(P[1],ε,ϱ,Qs[i])

    end

    for i in ProgressBar(eachindex(Qs))
        if plot_ball_Qs[i] == 1
            mean, width = mean_and_std(Qs[i])
            scatter!([mean], [width], color=:blue, markersize=4, markerstrokewidth=0.0, labels=nothing,)

        end

        if plot_intersection_Qs[i] == 1
            mean, width = mean_and_std(Qs[i])
            scatter!([mean], [width], color=:orange, markersize=4, markerstrokewidth=0.0, labels=nothing,)

        end

    end
end



display(plt)

