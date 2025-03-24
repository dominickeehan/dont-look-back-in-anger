# Dominic Keehan : 2025

function SES_weights(history_of_observations, α)

    T = length(history_of_observations)

    weights = [α*(1-α)^(t-1) for t in T:-1:1]
    weights .= weights/sum(weights)

    return weights
end

function windowing_weights(history_of_observations, window_size)

    T = length(history_of_observations)

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

function optimal_weights(history_of_observations, ρ_ϵ)

    T = length(history_of_observations)

    ρ, ϵ = ρ_ϵ

    Problem = Model(Ipoptimizer)

    @variables(Problem, begin
                            1>= w[t=1:T] >=0 
                        end)

    @constraint(Problem, sum(w[t] for t in 1:T) == 1)
    for t in 1:T-1
        @constraint(Problem, w[t] >= w[t+1])
    end

    @objective(Problem, Max, (1/(sum(w[t]^2 for t in 1:T)))*(max(ϵ-sum(w[t]*t for t in 1:T)*ρ,0))^2)

    optimize!(Problem)

    return [max(value(w[t]),0) for t in T:-1:1]

end



function triangular_weights(history_of_observations, θ_T)

    #T = length(history_of_observations)

    θ, T = θ_T

    weights = zeros(length(history_of_observations))

    for t in 1:T
        weights[t] = max(1-θ*t,0)
    end
    weights .= weights/sum(weights)

    return reverse(weights)

end