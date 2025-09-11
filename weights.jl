
function windowing_weights(T, window_size)

    weights = zeros(T)

    if window_size >= T
        weights .= 1
    else
        for t in T:-1:T-window_size+1
            weights[t] = 1
        end
    end

    weights = weights/sum(weights)

    return weights

end


function smoothing_weights(T, α)

    if α == 0; weights = zeros(T); weights .= 1/T; return weights; end
    
    weights = [α*(1-α)^(t-1) for t in T:-1:1]
    weights = weights/sum(weights)

    return weights

end


using JuMP, Ipopt

Ipoptimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

function W1_concentration_weights(T, ϱ╱ε)

    ε = 10

    ϱ = ϱ╱ε * ε

    if ϱ == 0; weights = zeros(T); weights .= 1/T; return weights; end

    if ϱ >= ε; weights = zeros(T); weights[T] = 1; return weights; end


    Problem = Model(Ipoptimizer)

    @variable(Problem, 1 >= w[t=1:T] >= 0)

    @constraint(Problem, sum(w[t] for t in 1:T) == 1)
    @constraint(Problem, sum(w[t]*(T-t+1)*ϱ for t in 1:T) <= ε)

    @objective(Problem, Max, (1/(sum(w[t]^2 for t in 1:T)))*(ε-(sum(w[t]*(T-t+1)*ϱ for t in 1:T)))^2)

    optimize!(Problem)
    #print(is_solved_and_feasible(Problem)) # Passes and solution looks good locally for ε = 10.

    weights = [max(value(w[t]),0) for t in 1:T]
    weights = weights/sum(weights)

    return weights

end


function W2_concentration_weights(T, ϱ╱ε)

    ε = 10

    ϱ = ϱ╱ε * ε

    if ϱ == 0; weights = zeros(T); weights .= 1/T; return weights; end

    if ϱ >= ε; weights = zeros(T); weights[T] = 1; return weights; end
    

    p = 2

    Problem = Model(Ipoptimizer)

    @variable(Problem, 1 >= w[t=1:T] >= 0)

    @constraint(Problem, sum(w[t] for t in 1:T) == 1)
    @constraint(Problem, sum(w[t]*(T-t+1)^p*ϱ^p for t in 1:T) <= ε^p)

    @objective(Problem, Max, (1/(sum(w[t]^2 for t in 1:T)))*((ε-(sum(w[t]*(T-t+1)^p*ϱ^p for t in 1:T))^(1/p))^(2*p)))

    optimize!(Problem)
    #print(is_solved_and_feasible(Problem)) # Passes and solution looks good locally for ε = 10.

    weights = [max(value(w[t]),0) for t in 1:T]
    weights = weights/sum(weights)

    return weights

end


function REMK_intersection_weights(K, ϱ╱ε) 

    return ones(K) * ϱ╱ε

end

function REMK_intersection_ball_radii(K, ε, ϱ╱ε) 
    
    ϱ = ϱ╱ε * ε

    return [ε+(K-k+1)*ϱ for k in 1:K]

end




function Wp_concentration_weights(p, T, ϱ╱ε)

    ε = 10

    ϱ = ϱ╱ε * ε

    if ϱ == 0; weights = zeros(T); weights .= 1/T; return weights; end

    if ϱ >= ε; weights = zeros(T); weights[T] = 1; return weights; end

    Problem = Model(Ipoptimizer)

    @variable(Problem, 1 >= w[t=1:T] >= 0)

    @constraint(Problem, sum(w[t] for t in 1:T) == 1)
    @constraint(Problem, sum(w[t]*(T-t+1)^p*ϱ^p for t in 1:T) <= ε^p)

    @objective(Problem, Max, (1/(sum(w[t]^2 for t in 1:T)))*((ε-(sum(w[t]*(T-t+1)^p*ϱ^p for t in 1:T))^(1/p))^(2*p)))

    optimize!(Problem)
    #print(is_solved_and_feasible(Problem)) # Passes and solution looks good locally for ε = 10.


    weights = [max(value(w[t]),0) for t in 1:T]
    weights = weights/sum(weights)

    return weights

end