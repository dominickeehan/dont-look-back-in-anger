using JuMP, Ipopt

Ipoptimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)


function Wp_weights(p, T, ρ╱ε)

    ε = 10.0

    ρ = ρ╱ε * ε

    if ρ == 0.0; weights = zeros(T); weights .= 1/T; return weights; end

    if ρ >= ε; weights = zeros(T); weights[T] = 1.0; return weights; end

    Problem = Model(Ipoptimizer)

    @variable(Problem, 1.0 >= w[t=1:T] >= 0.0)

    @constraint(Problem, sum(w[t] for t in 1:T) == 1.0)
    @constraint(Problem, sum(w[t]*(T-t+1)^p for t in 1:T)*ρ^p <= ε^p)

    @objective(Problem, Max, (1/(sum(w[t]^2 for t in 1:T)))*((ε-(sum(w[t]*(T-t+1)^p for t in 1:T))^(1/p)*ρ)^(2*p)))

    optimize!(Problem)

    #display(is_solved_and_feasible(Problem)) # Passes and solution looks good locally for ε = 10.

    weights = max.(value.(w),0.0)
    weights = weights/sum(weights)

    return weights
end


function W2_weights(T, ρ╱ε)

    return Wp_weights(2, T, ρ╱ε)
end


function REMK_intersection_weights(K, ρ╱ε) 

    return ones(K) * ρ╱ε
end


function REMK_intersection_ball_radii(K, ε, ρ╱ε) 
    
    ρ = ρ╱ε * ε

    return [ε+(K-k+1)*ρ for k in 1:K]
end


function windowing_weights(T, window_size)

    weights = zeros(T)

    if window_size >= T
        weights .= 1.0
    else
        for t in T:-1:T-window_size+1
            weights[t] = 1.0
        end
    end

    weights = weights/sum(weights)

    return weights
end


function smoothing_weights(T, α)

    if α == 0.0; weights = zeros(T); weights .= 1/T; return weights; end
    
    weights = [α*(1-α)^(t-1) for t in T:-1:1]
    weights = weights/sum(weights)

    return weights
end

