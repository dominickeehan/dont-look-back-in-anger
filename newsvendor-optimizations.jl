using Statistics, StatsBase
using JuMP, MathOptInterface, Gurobi

number_of_dimensions = 3

# Per-dimension problem parameters
initial_demand_probability = 1/3
number_of_consumers = 1000
cu = 4 # Per-unit underage cost.
co = 1 # Per-unit overage cost.

env = Gurobi.Env()
GRBsetintparam(env, "OutputFlag", 0)
optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

# Construct .
choices = (cu, -co)
iterators = Iterators.product(ntuple(_ -> choices, number_of_dimensions)...)
a = vec([collect(iterator) for iterator in iterators])
choices = (-cu, co)
iterators = Iterators.product(ntuple(_ -> choices, number_of_dimensions)...)
b = vec([collect(iterator) for iterator in iterators])

C = zeros((2*number_of_dimensions, number_of_dimensions))
g = zeros(2*number_of_dimensions)
for i in 1:number_of_dimensions
    C[2*i-1, i] = -1.0
    C[2*i, i] = 1.0
    g[2*i-1] = 0.0
    g[2*i] = number_of_consumers

end


function SO_newsvendor_objective_value_and_order(_, demands, weights, doubling_count) 

    nonzero_weight_indices = weights .> 0
    weights = weights[nonzero_weight_indices]
    weights = weights/sum(weights)
    demands = demands[nonzero_weight_indices]

    T = length(demands)

    Problem = Model(optimizer)

    @variables(Problem, begin
                            number_of_consumers >= order[i=1:number_of_dimensions] >= 0
                                 s[t=1:T]
                        end)

    for t in 1:T
        for l in eachindex(a)
            @constraints(Problem, begin
                                        b[l]'*order + a[l]'*demands[t] <= s[t]
                                  end)
        end
    end

    @objective(Problem, Min, weights'*s)

    optimize!(Problem)

    if is_solved_and_feasible(Problem)
        return objective_value(Problem), [value(order[i]) for i in 1:number_of_dimensions], doubling_count 

    else
        #order = quantile(demands, Weights(weights), Cu/(Co+Cu))
        #return sum(weights[t] * (Cu*max(demands[t]-order,0) + Co*max(order-demands[t],0)) for t in eachindex(weights)), order, doubling_count
        throw("throw")

    end
end

function W2_newsvendor_objective_value_and_order(ε, demands, weights, doubling_count) 

    if ε == 0; return SO_newsvendor_objective_value_and_order(ε, demands, weights, doubling_count); end

    nonzero_weight_indices = weights .> zero_weight_tolerance
    weights = weights[nonzero_weight_indices]
    weights = weights/sum(weights)
    demands = demands[nonzero_weight_indices]

    T = length(demands)

    Problem = Model(optimizer)

    @variables(Problem, begin
                            number_of_consumers >= order >= 0
                                λ >= 0
                                γ[t=1:T]
                                z[t=1:T,l=1:2,m=1:2] >= 0
                                w[t=1:T,l=1:2,i=1:number_of_dimensions]
                        end)

    for t in 1:T
        for l in eachindex(a)
            @constraints(Problem, begin
                                        # b[l]'*order + w[t,l,:]'*demands[t] + (1/4)*(1/λ)*w[t,i]^2 + z[t,i,:]'*g <= γ[t] 
                                        # <==> w[t,i]^2 <= 2*(2*λ)*(γ[t] - b(order)[i] - w[t,i]*demands[t] - z[t,i,:]'*g) 
                                        # <==>
                                        [2*λ; γ[t] - b(order)[i] - w[t,i]*demands[t] - z[t,i,:]'*d; w[t,i]] in MathOptInterface.RotatedSecondOrderCone(3)
                                        a[i] - C'*z[t,i,:] == w[t,i]
                                end)
        end
    end

    @objective(Problem, Min, (ε^2)*λ + weights'*γ)

    optimize!(Problem)

    # Check the problem is solved and feasible.
    if is_solved_and_feasible(Problem)
        return objective_value(Problem), value(order), doubling_count
    
    else # Attempt a high precision solve otherwise.
        set_attribute(Problem, "BarHomogeneous", 1)
        set_attribute(Problem, "NumericFocus", 3)
        optimize!(Problem)

        # Try to return a suboptimal solution from a possibly early termination as the problem is always feasible and bounded. 
        # (This may be neccesary due to near infeasiblity after convex reformulation.)
        try
            return objective_value(Problem), value(order), doubling_count
    
        catch # As a last resort, double the ambiguity radius and try again.
            return W2_newsvendor_objective_value_and_order(2*ε, demands, weights, doubling_count+1)

        end
    end
end

function REMK_intersection_W2_newsvendor_objective_value_and_order(ε, demands, weights, doubling_count)

    K = length(demands)

    ball_radii = REMK_intersection_ball_radii(K, ε, weights[end])

    # Check if the intersection of balls is (nearly) empty, and scale up the radii if so.
    L = demands .- ball_radii
    U = demands .+ ball_radii

    # Indices of worst lower and upper endpoints.
    i_maxL = argmax(L) # Index attaining max lower bound.
    j_minU = argmin(U) # Index attaining min upper bound.

    empty_intersection_ratio = (demands[i_maxL] - demands[j_minU]) / (ball_radii[i_maxL] + ball_radii[j_minU])

    if empty_intersection_ratio >= 1.0 # Scale to sufficiently nonempty.
        demand = demands[i_maxL] - empty_intersection_ratio*ball_radii[i_maxL]
        return SO_newsvendor_objective_value_and_order(0.0, demand, 1.0, doubling_count)

        #ball_radii = (empty_intersection_ratio / empty_intersection_ratio_tolerance) * ball_radii

    end

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
    #if is_solved_and_feasible(Problem) 
    #    return objective_value(Problem), value(order), doubling_count
    
    #else
    #    throw=throw
        #return REMK_intersection_W2_newsvendor_objective_value_and_order(2*ε, demands, weights, doubling_count+1)
    
    #end

    # Check the problem is solved and feasible.
    if is_solved_and_feasible(Problem)
        return objective_value(Problem), value(order), doubling_count
    
    else # Attempt a high precision solve otherwise.
        #set_optimizer(Problem, high_precision_optimizer); optimize!(Problem)
        set_attribute(Problem, "BarHomogeneous", 1); set_attribute(Problem, "NumericFocus", 3)
        optimize!(Problem)

        # Try to return a suboptimal solution from a possibly early termination as the problem is always feasible and bounded. 
        # (This may be neccesary due to near infeasiblity after convex reformulation caused by very unbalanced weights.)
        try #is_solved_and_feasible(Problem)
            return objective_value(Problem), value(order), doubling_count
    
        catch # As a last resort, double the scaled-up ambiguity radius and try again.
            return REMK_intersection_W2_newsvendor_objective_value_and_order(2*max(empty_intersection_ratio, 1)*ε, demands, weights, doubling_count+1)

        end
    end

end

5
