using JuMP, MathOptInterface, Gurobi

D = 10000

Cu = 4 # Per-unit underage cost.
Co = 1 # Per-unit overage cost.


function SO_newsvendor_value_and_order(_, demands, weights) 
    
    order = quantile(demands, Weights(weights), Cu/(Co+Cu))
    
    expected_underage = sum(weights[t]*max(demands[t] - order,0) for t in eachindex(weights))
    expected_overage = sum(weights[t]*max(order - demands[t],0) for t in eachindex(weights))

    return Cu*expected_underage + Co*expected_overage, order

end


env = Gurobi.Env() 
GRBsetintparam(env, "OutputFlag", 0)
optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

function W1_newsvendor_value_and_order(ε, demands, weights) 

    if ε == 0; return SO_newsvendor_value_and_order(ε, demands, weights); end


    demands = demands[weights .>= 1e-6]
    weights = weights[weights .>= 1e-6]
    weights .= weights/sum(weights)

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

    try; return objective_value(Problem), value(order); catch; return W1_newsvendor_value_and_order(2*ε, demands, weights); end

end


BarHomogeneous_env = Gurobi.Env() 
GRBsetintparam(BarHomogeneous_env, "OutputFlag", 0)
GRBsetintparam(BarHomogeneous_env, "BarHomogeneous", 1)
BarHomogeneous_optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(BarHomogeneous_env))

function W2_newsvendor_value_and_order(ε, demands, weights) 

    if ε == 0; return SO_newsvendor_value_and_order(ε, demands, weights); end


    demands = demands[weights .>= 1e-6] # 1.5 mins at 1e-6 vs 1 min at 1e-3.
    weights = weights[weights .>= 1e-6]
    weights .= weights/sum(weights)

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

    try; return objective_value(Problem), value(order); catch; return W2_newsvendor_value_and_order(2*ε, demands, weights); end

end


REMK_intersection_ball_radii(K, ε, ϱ) = [ε+(K-k)*ϱ for k in 1:K]

function REMK_intersection_based_W2_newsvendor_value_and_order(_, demands, ball_radii)

    K = length(ball_radii)
    demands = demands[end-K+1:end]


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

    try; return objective_value(Problem), value(order); catch; return REMK_intersection_based_W2_newsvendor_value_and_order(0, demands, 2*ball_radii); end

end


#include("weights.jl")

#display(REMK_intersection_based_W2_newsvendor_value_and_order(0, [1, 2, 3, 4, 100], REMK_intersection_ball_radii(5, 100000000, 0)))
#W2_newsvendor_value_and_order(10000000, [1, 2, 3, 4, 5], ones(5)*1/5)
#W1_newsvendor_value_and_order(1000000, [1, 2, 3, 4, 5], ones(5)*1/5)