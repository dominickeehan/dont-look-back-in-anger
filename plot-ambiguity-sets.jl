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



support = LinRange(-120,120,1000)
number_of_points = 20
bin_div = 2

ε = 40
ϱ = ε/10

points = Vector(LinRange(-50,50,number_of_points))
weights = W2_concentration_weights(number_of_points, ϱ/ε)
P = [points, weights]

number_of_distributions = 100000 #1000000

Qs = [[zeros(number_of_points), zeros(number_of_points)] for _ in 1:number_of_distributions]
Threads.@threads for i in ProgressBar(eachindex(Qs))
    local n = number_of_points #rand(50:number_of_points)
    local points = rand(support, n)
    local weights = rand(Uniform(0,1), n)
    local weights .= normalise(weights)

    Qs[i] = [points, weights]

end


plot_ball_Qs = zeros(length(Qs))

Threads.@threads for i in ProgressBar(eachindex(Qs))
    plot_ball_Qs[i] = in_W2_ball(P,ε,Qs[i])

end





alpha = 0.0
fillalpha = max(1/sum(plot_ball_Qs),0.002)
display(sum(plot_ball_Qs))
plt = plot(xlims=(support[1],support[end]))

for i in ProgressBar(eachindex(Qs))
    if plot_ball_Qs[i] == 1
        stephist!(Qs[i][1], weights=Qs[i][2], bins=ceil(Int,length(Qs[i][1])/bin_div), normalise=true, color=:red, alpha=alpha, fill=true, fillalpha=fillalpha, labels=nothing,)

    end

end

#stephist!(P[1], weights=P[2], bins=ceil(Int,length(P[1])/bin_div), normalise=true, color=:black, alpha=1.0, fill=true, fillalpha=0.0, labels=nothing,)
plot!(P[1], P[2], seriestype=:sticks, color=:black, linewidth=1, markershape=:circle, markersize=2, markercolor=:grey, labels=nothing,)
title!("Concentration \$ε=$ε\$, \$ϱ=$ϱ\$")
ylims!((0,1))
display(plt)



plot_intersection_Qs = zeros(length(Qs))

Threads.@threads for i in ProgressBar(eachindex(Qs))
    plot_intersection_Qs[i] = in_W2_intersection(points,ε,ϱ,Qs[i])

end

fillalpha = max(1/sum(plot_intersection_Qs),0.002)
display(sum(plot_intersection_Qs))
plt = plot(xlims=(support[1],support[end]))

for i in ProgressBar(eachindex(Qs))
    if plot_intersection_Qs[i] == 1
        stephist!(Qs[i][1], weights=Qs[i][2], bins=ceil(Int,length(Qs[i][1])/bin_div), normalise=true, color=:red, alpha=alpha, fill=true, fillalpha=fillalpha, labels=nothing,)

    end

end

#plot!(P[1], 1/number_of_points*ones(number_of_points), seriestype=:sticks, color=:black, linewidth=1, markershape=:circle, markersize=2, markercolor=:grey, labels=nothing,)
title!("Intersection \$ε=$ε\$, \$ϱ=$ϱ\$")
ylims!((0,1))
display(plt)