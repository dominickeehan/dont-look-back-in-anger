# Dominic Keehan : 2025

function smoothing_weights(T, ε, α)

    weights = [α*(1-α)^(t-1) for t in T:-1:1]
    weights .= weights/sum(weights)

    return weights
end

function windowing_weights(T, ε, window_size)

    weights = zeros(T)

    if window_size >= T
        weights .= 1
    else
        for t in T:-1:T-(window_size-1)
            weights[t] = 1
        end
    end

    weights .= weights/sum(weights)

    return weights
end

using JuMP, Ipopt

Ipoptimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)#, "tol" => 1e-9)

function W₁_concentration_weights(T, ε, ϱ)

    p = 1

    if ϱ >= (1/1)*ε; ϱ = (1/1)*ε; end

    Problem = Model(Ipoptimizer)

    @variable(Problem, 1>= w[t=1:T] >=0)

    @constraint(Problem, sum(w[t] for t in 1:T) == 1)
    @constraint(Problem, sum(w[t]*t^p*ϱ^p for t in 1:T) <= ε^p)

    @objective(Problem, Max, (1/(sum(w[t]^2 for t in 1:T)))*((ε-(sum(w[t]*t^p*ϱ^p for t in 1:T))^(1/p))^(2*p)))

    optimize!(Problem)

    weights = [max(value(w[t]),0) for t in 1:T]
    weights .= weights/sum(weights)

    return reverse(weights)

end

function W₂_concentration_weights(T, ε, ϱ)

    p = 2

    if ϱ >= (1/1)*ε; ϱ = (1/1)*ε; end

    Problem = Model(Ipoptimizer)

    @variable(Problem, 1>= w[t=1:T] >=0)

    @constraint(Problem, sum(w[t] for t in 1:T) == 1)
    @constraint(Problem, sum(w[t]*t^p*ϱ^p for t in 1:T) <= ε^p)

    @objective(Problem, Max, (1/(sum(w[t]^2 for t in 1:T)))*((ε-(sum(w[t]*t^p*ϱ^p for t in 1:T))^(1/p))^(2*p)))

    optimize!(Problem)

    weights = [max(value(w[t]),0) for t in 1:T]
    weights .= weights/sum(weights)

    return reverse(weights)

end