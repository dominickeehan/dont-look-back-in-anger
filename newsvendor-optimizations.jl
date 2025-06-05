using Statistics, StatsBase
using JuMP, MathOptInterface, Gurobi

D = 10000 # 10000

Cu = 4 # Per-unit underage cost.
Co = 1 # Per-unit overage cost.


env = Gurobi.Env()
GRBsetintparam(env, "OutputFlag", 0)
optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

function SO_newsvendor_value_and_order(_, demands, weights, doubling_count) 

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
    
        expected_underage = sum(weights[t]*max(demands[t] - order,0) for t in eachindex(weights))
        expected_overage = sum(weights[t]*max(order - demands[t],0) for t in eachindex(weights))

        return Cu*expected_underage + Co*expected_overage, order, doubling_count+1
    
    end

end


function W1_newsvendor_value_and_order(ε, demands, weights, doubling_count) 

    if ε == 0; return SO_newsvendor_value_and_order(ε, demands, weights, doubling_count); end


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

    @objective(Problem, Min, ε*λ + weights'*s)

    optimize!(Problem)

    try
        return objective_value(Problem), value(order), doubling_count
    
    catch
        return W1_newsvendor_value_and_order(2*ε, demands, weights, doubling_count+1)
    
    end

end


BarHomogeneous_env = Gurobi.Env() 
GRBsetintparam(BarHomogeneous_env, "OutputFlag", 0)
GRBsetintparam(BarHomogeneous_env, "BarHomogeneous", 1)
BarHomogeneous_optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(BarHomogeneous_env))

function W2_newsvendor_value_and_order(ε, demands, weights, doubling_count) 

    if ε == 0; return SO_newsvendor_value_and_order(ε, demands, weights, doubling_count); end


    T = length(demands)

    Problem = Model(BarHomogeneous_optimizer)

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

    @objective(Problem, Min, ε*λ + weights'*γ)

    optimize!(Problem)

    try
        return objective_value(Problem), value(order), doubling_count
    
    catch
        return W2_newsvendor_value_and_order(2*ε, demands, weights, doubling_count+1)
    
    end

end


function REMK_intersection_weights(K, ϱ_divided_by_ε) 

    return ones(K)*ϱ_divided_by_ε

end

function REMK_intersection_ball_radii(K, ε, ϱ_divided_by_ε) 
    
    ϱ = ϱ_divided_by_ε * ε

    return [ε+(K-k+1)*ϱ for k in 1:K]

end

function REMK_intersection_W2_newsvendor_value_and_order(ε, demands, weights, doubling_count)

    K = length(demands)

    ball_radii = REMK_intersection_ball_radii(K, ε, weights[end])

    Problem = Model(BarHomogeneous_optimizer)

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
                                s[i=1:2,k=1:K]
                        end)

    for i in 1:2
        @constraints(Problem, begin
                                    # b(order)[i] + sum(w[i,k]*demands[k] + (1/4)*(1/λ[k])*w[i,k]^2 for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K)
                                    # <==> b(order)[i] + sum(w[i,k]*demands[k] + s[i,k] for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K)
                                    # (1/4)*(1/λ[K])*w[i,k]^2 <= s[i,k] for all i,k,
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

    @objective(Problem, Min, sum(ball_radii[k]*λ[k] for k in 1:K) + sum(γ[k] for k in 1:K))

    optimize!(Problem)

    try
        return objective_value(Problem), value(order), doubling_count
    
    catch
        return REMK_intersection_W2_newsvendor_value_and_order(2*ε, demands, weights, doubling_count+1)
    
    end

end

