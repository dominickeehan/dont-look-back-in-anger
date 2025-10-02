using Statistics, StatsBase
using JuMP, MathOptInterface, Gurobi

initial_demand_probability = 0.1
D = 10000 # Number of consumers.

nonzero_weight_tolerance = 0

Cu = 4 # Per-unit underage cost.
Co = 1 # Per-unit overage cost.


env = Gurobi.Env()
GRBsetintparam(env, "OutputFlag", 0)
GRBsetintparam(env, "BarHomogeneous", 1) # Useful for dealing with very unbalanced weights/intersections giving nearly infeasible problems.
#GRBsetintparam(env, "NumericFocus", 3) # Useful for dealing with very unbalanced weights.

optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

function SO_newsvendor_objective_value_and_order(_, demands, weights, doubling_count) 

    nonzero_weight_indices = weights .> nonzero_weight_tolerance
    weights = weights[nonzero_weight_indices]
    weights = weights/sum(weights)
    demands = demands[nonzero_weight_indices]

    T = length(demands)

    Problem = Model(optimizer)

    C = [-1; 1]
    d = [0, D]
    a = [Cu,-Co]
    b(order) = [-Cu*order, Co*order]

    @variables(Problem, begin
                            D >= order >= 0
                                 s[t=1:T]
                        end)

    for t in 1:T
        for i in 1:2
            @constraints(Problem, begin
                                        b(order)[i] + a[i]*demands[t] <= s[t]
                                  end)
        end
    end

    @objective(Problem, Min, weights'*s)

    optimize!(Problem)

    try
        return objective_value(Problem), value(order), doubling_count 

    catch
        order = quantile(demands, Weights(weights), Cu/(Co+Cu))

        return sum(weights[t] * (Cu*max(demands[t]-order,0) + Co*max(order-demands[t],0)) for t in eachindex(weights)), order, doubling_count+1
    
    end
end


function W1_newsvendor_objective_value_and_order(ε, demands, weights, doubling_count) 

    if ε == 0; return SO_newsvendor_objective_value_and_order(ε, demands, weights, doubling_count); end

    nonzero_weight_indices = weights .> nonzero_weight_tolerance
    weights = weights[nonzero_weight_indices]
    weights = weights/sum(weights)
    demands = demands[nonzero_weight_indices]

    T = length(demands)

    Problem = Model(optimizer)

    C = [-1; 1]
    d = [0, D]
    a = [Cu,-Co]
    b(order) = [-Cu*order, Co*order]

    @variables(Problem, begin
                            D >= order >= 0
                                 λ
                                 s[t=1:T]
                                 γ[t=1:T,i=1:2,j=1:2] >= 0
                                 z[t=1:T,i=1:2]
                        end)

    for t in 1:T
        for i in 1:2
            @constraints(Problem, begin
                                        b(order)[i] + a[i]*demands[t] + γ[t,i,:]'*(d-C*demands[t]) <= s[t]
                                        z[t,i] <= λ
                                        C'*γ[t,i,:] - a[i] <= z[t,i]
                                       -C'*γ[t,i,:] + a[i] <= z[t,i]
                                  end)
        end
    end

    @objective(Problem, Min, (ε^1)*λ + weights'*s)

    optimize!(Problem)

    # Try to return a suboptimal solution from an early termination as the problem is always feasible.
    # (This may be neccesary due to near infeasiblity caused by very unbalanced weights.)
    try
        return objective_value(Problem), value(order), doubling_count
    
    catch
        return W1_newsvendor_objective_value_and_order(2*ε, demands, weights, doubling_count+1)
    
    end
end

function W2_newsvendor_objective_value_and_order(ε, demands, weights, doubling_count) 

    if ε == 0; return SO_newsvendor_objective_value_and_order(ε, demands, weights, doubling_count); end

    nonzero_weight_indices = weights .> nonzero_weight_tolerance
    weights = weights[nonzero_weight_indices]
    weights = weights/sum(weights)
    demands = demands[nonzero_weight_indices]

    T = length(demands)

    Problem = Model(optimizer)

    C = [-1; 1]
    d = [0, D]
    a = [Cu,-Co]
    b(order) = [-Cu*order, Co*order]

    @variables(Problem, begin
                            D >= order >= 0
                                λ >= 0
                                γ[t=1:T]
                                z[t=1:T,i=1:2,j=1:2] >= 0
                                w[t=1:T,i=1:2]
                        end)

    for t in 1:T
        for i in 1:2
            @constraints(Problem, begin
                                        # b(order)[i] + w[t,i]*demands[t] + (1/4)*(1/λ)*w[t,i]^2 + z[t,i,:]'*d <= γ[t] 

                                        # <==> w[t,i]^2 <= 2*(2*λ)*(γ[t] - b(order)[i] - w[t,i]*demands[t] - z[t,i,:]'*d) 
                                        
                                        # <==>
                                        [2*λ; γ[t] - b(order)[i] - w[t,i]*demands[t] - z[t,i,:]'*d; w[t,i]] in MathOptInterface.RotatedSecondOrderCone(3)
                                        
                                        a[i] - C'*z[t,i,:] == w[t,i]
                                end)
        end
    end

    @objective(Problem, Min, (ε^2)*λ + weights'*γ)

    optimize!(Problem)

    # Try to return a suboptimal solution from an early termination as the problem is always feasible.
    # (This may be neccesary due to near infeasiblity after convex reformulation caused by very unbalanced weights.)
    try
        return objective_value(Problem), value(order), doubling_count
    
    catch
        return W2_newsvendor_objective_value_and_order(2*ε, demands, weights, doubling_count+1)
        #return Inf, D, 1

    end
end


function REMK_intersection_W2_newsvendor_objective_value_and_order(ε, demands, weights, doubling_count)

    K = length(demands)

    ball_radii = REMK_intersection_ball_radii(K, ε, weights[end])

    Problem = Model(optimizer)

    C = [-1; 1]
    d = [0, D]
    a = [Cu,-Co]
    b(order) = [-Cu*order, Co*order]

    @variables(Problem, begin
                            D >= order >= 0
                                λ[k=1:K] >= 0
                                γ[k=1:K]
                                z[i=1:2,j=1:2] >= 0
                                w[i=1:2,k=1:K]
                                s[i=1:2,k=1:K] >= 0
                        end)

    for i in 1:2
        @constraints(Problem, begin
                                    # b(order)[i] + sum(w[i,k]*demands[k] + (1/4)*(1/λ[k])*w[i,k]^2 for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K)
                                    
                                    # <==> b(order)[i] + sum(w[i,k]*demands[k] + s[i,k] for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K)
                                    # 0 <= (1/4)*(1/λ[K])*w[i,k]^2 <= s[i,k] for all i,k <==> w[i,k]^2 <= 2*(2*λ[K])*s[i,k] for all i,k,
                                    
                                    # <==> b(order)[i] + sum(w[i,k]*demands[k] + s[i,k] for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K)
                                    # [2*λ[k]; s[i,k]; w[i,k]] in MathOptInterface.RotatedSecondOrderCone(3) for all i,k,
                                    
                                    # <==>
                                    b(order)[i] + sum(w[i,k]*demands[k] + s[i,k] for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K)
                                    a[i] - C'*z[i,:] == sum(w[i,k] for k in 1:K)
                                end)

        for k in 1:K
            @constraints(Problem, begin
                                        [2*λ[k]; s[i,k]; w[i,k]] in MathOptInterface.RotatedSecondOrderCone(3) 
                                    end)
        end
    end

    @objective(Problem, Min, sum((ball_radii[k]^2)*λ[k] for k in 1:K) + sum(γ[k] for k in 1:K))

    optimize!(Problem)

    # Feasibility check to ensure intersection is nonempty before returning. 
    # (Otherwise, occasionally BarHomogeneous will return a crazy solution 
    # from an early termination since the problem is actually infeasible.)
    if is_solved_and_feasible(Problem) 
        return objective_value(Problem), value(order), doubling_count
    
    else
        return REMK_intersection_W2_newsvendor_objective_value_and_order(2*ε, demands, weights, doubling_count+1)
    
    end
end
