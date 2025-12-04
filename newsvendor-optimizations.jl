using LinearAlgebra
using Statistics, StatsBase
using JuMP, MathOptInterface, Gurobi

env = Gurobi.Env()
GRBsetintparam(env, "OutputFlag", 0)
optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

# Construct problem matrors.
# See "Wasserstein Distributionally Robust Optimization with Heterogeneous Data Sources" 
# by Yves Rychener, Adrián Esteban-Pérez, Juan M. Morales, and Daniel Kuhn (arXiv:2407.13582v2),
# Corollary 2.
choices = (cu, -co) # We assume that the underage and overage costs of each good are the same.
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
        return objective_value(Problem), value.(order), doubling_count 

    else
        #order = quantile(demands, Weights(weights), Cu/(Co+Cu))
        #return sum(weights[t] * (Cu*max(demands[t]-order,0) + Co*max(order-demands[t],0)) for t in eachindex(weights)), order, doubling_count
        throw("throw")

    end
end


function W2_newsvendor_objective_value_and_order(ε, demands, weights, doubling_count) 

    if ε == 0; return SO_newsvendor_objective_value_and_order(ε, demands, weights, doubling_count); end

    nonzero_weight_indices = weights .> 0
    weights = weights[nonzero_weight_indices]
    weights = weights/sum(weights)
    demands = demands[nonzero_weight_indices]

    T = length(demands)

    # See "Wasserstein Distributionally Robust Optimization with Heterogeneous Data Sources" 
    # by Yves Rychener, Adrián Esteban-Pérez, Juan M. Morales, and Daniel Kuhn (arXiv:2407.13582v2),
    # Corollary 2.
    Problem = Model(optimizer)

    @variables(Problem, begin
                            number_of_consumers >= order[i=1:number_of_dimensions] >= 0
                            λ >= 0
                            γ[t=1:T]
                            z[t=1:T,l=1:length(a),m=1:length(g)] >= 0
                            w[t=1:T,l=1:length(a),i=1:number_of_dimensions]
                        end)

    for t in 1:T
        for l in eachindex(a)
            @constraints(Problem, begin
                                    # b[l]'*order + w[t,l,:]'*demands[t] + (1/4)*(1/λ)*sum(w[t,l,i]^2 for i in 1:number_of_dimensions) + z[t,l,:]'*g <= γ[t] 
                                    # <==> sum(w[t,l,i]^2 for i in 1:number_of_dimensions) <= 2*(2*λ)*(γ[t] - b[l]'*order - w[t,l,:]'*demands[t] - z[t,l,:]'*g) 
                                    # <==>
                                    [2*λ; γ[t] - b[l]'*order - w[t,l,:]'*demands[t] - z[t,l,:]'*g; w[t,l,:]] in MathOptInterface.RotatedSecondOrderCone(2+number_of_dimensions)
                                    a[l] .- C'*z[t,l,:] .== w[t,l,:]
                                  end)
        end
    end

    @objective(Problem, Min, (ε^2)*λ + weights'*γ)

    set_attribute(Problem, "BarHomogeneous", -1)
    set_attribute(Problem, "NumericFocus", 0)
    optimize!(Problem)

    # Check the problem is solved and feasible.
    if is_solved_and_feasible(Problem)
        return objective_value(Problem), value.(order), doubling_count
    
    else # Attempt a high precision solve otherwise.
        set_attribute(Problem, "BarHomogeneous", 1)
        set_attribute(Problem, "NumericFocus", 3)
        optimize!(Problem)

        # Try to return a suboptimal solution from a possibly early termination as the problem is always feasible and bounded. 
        # (This may be neccesary due to near infeasiblity after convex reformulation.)
        try
            return objective_value(Problem), value.(order), doubling_count
    
        catch # As a last resort, double the ambiguity radius and try again.
            return W2_newsvendor_objective_value_and_order(2*ε, demands, weights, doubling_count+1)

        end
    end
end


function REMK_intersection_W2_newsvendor_objective_value_and_order(ε, demands, weights, doubling_count)

    K = length(demands)

    ball_radii = REMK_intersection_ball_radii(K, ε, weights[end])

    # In the following distributional ball-intersection feasibility problem the equivalence to working in R^m follows since 
    # W₂(P,1_ξ) = sqrt(sum((E(P)_i-ξ_i)^2) + Tr(Cov(P))) which drives Tr(Cov(P)) -> 0 and P -> 1_E(P) at extremal distributions.  
    Ball_Intersection_Feasibility_Problem = Model(optimizer)

    @variables(Ball_Intersection_Feasibility_Problem, begin
                                                        number_of_consumers >= x[i=1:number_of_dimensions] >= 0 # Feasible intersection.
                                                        λ >= 0 # Radii up-scaling factor.
                                                      end)

    for k in 1:K
        @constraint(Ball_Intersection_Feasibility_Problem,
                        # ‖x - ball_centers[k]‖₂ <= λ*ball_radii[k] for all k
                        # <==>
                        [1/2*ball_radii[k]*λ; ball_radii[k]*λ; x - demands[k]] in MathOptInterface.RotatedSecondOrderCone(2+number_of_dimensions))
    end

    @objective(Ball_Intersection_Feasibility_Problem, Min, λ) # Minimize the up-scaling factor required for a feasible intersection.

    set_attribute(Ball_Intersection_Feasibility_Problem, "BarHomogeneous", -1)
    set_attribute(Ball_Intersection_Feasibility_Problem, "NumericFocus", 0)
    optimize!(Ball_Intersection_Feasibility_Problem)

    if is_solved_and_feasible(Ball_Intersection_Feasibility_Problem)
        if value(λ) >= 1 # Then we had to scale up the ball radii for the ambiguity set to be nonempty.
            # At the point of scaling where the set first becomes nonempty, the only distribution is the point-mass distribution 
            # at the point where all the radii touch, i.e., at the solution to the ball-intersection feasibility problem. 
            return SO_newsvendor_objective_value_and_order(0.0, [value.(x)], [1.0], doubling_count)

        end
    else
        throw("throw")

    end

    # See "Wasserstein Distributionally Robust Optimization with Heterogeneous Data Sources" 
    # by Yves Rychener, Adrián Esteban-Pérez, Juan M. Morales, and Daniel Kuhn (arXiv:2407.13582v2),
    # Corollary 2.
    Problem = Model(optimizer)

    @variables(Problem, begin
                            number_of_consumers >= order[i=1:number_of_dimensions] >= 0
                            λ[k=1:K] >= 0
                            γ[k=1:K]
                            z[l=1:length(a),m=1:length(g)] >= 0
                            w[l=1:length(a),k=1:K,i=1:number_of_dimensions]
                            s[l=1:length(a),k=1:K] >= 0
                        end)

    for l in eachindex(a)
        @constraints(Problem, begin
                                # b[l]'*order + sum(w[l,k,:]'*demands[k] + (1/4)*(1/λ[k])*sum(w[l,k,i]^2 for i in 1:number_of_dimensions) for k in 1:K) + z[l,:]'*g <= sum(γ[k] for k in 1:K)
                                # <==> b[l]'*order + sum(w[l,k,:]'*demands[k] + s[l,k] for k in 1:K) + z[l,:]'*g <= sum(γ[k] for k in 1:K),
                                # 0 <= (1/4)*(1/λ[K])*sum(w[l,k,i]^2 for i in 1:number_of_dimensions) <= s[l,k] for all l,k <==> sum(w[l,k,i]^2 for i in 1:number_of_dimensions) <= 2*(2*λ[K])*s[l,k] for all l,k,
                                # <==> b[l]'*order + sum(w[l,k,:]'*demands[k] + s[l,k] for k in 1:K) + z[l,:]'*g <= sum(γ[k] for k in 1:K),
                                # [2*λ[k]; s[l,k]; w[l,k,:]] in MathOptInterface.RotatedSecondOrderCone(2+number_of_dimensions) for all l,k,       
                                # <==>
                                b[l]'*order + sum(w[l,k,:]'*demands[k] + s[l,k] for k in 1:K) + z[l,:]'*g <= sum(γ[k] for k in 1:K)
                                a[l] .- C'*z[l,:] .== sum(w[l,k,:] for k in 1:K)
                              end)

        for k in 1:K
            @constraints(Problem, begin
                                    [2*λ[k]; s[l,k]; w[l,k,:]] in MathOptInterface.RotatedSecondOrderCone(2+number_of_dimensions) 
                                  end)

        end
    end

    @objective(Problem, Min, sum((ball_radii[k]^2)*λ[k] for k in 1:K) + sum(γ[k] for k in 1:K))

    set_attribute(Problem, "BarHomogeneous", -1)
    set_attribute(Problem, "NumericFocus", 0)
    optimize!(Problem)

    # Check the problem is solved and feasible.
    if is_solved_and_feasible(Problem)
        return objective_value(Problem), value.(order), doubling_count
    
    else # Attempt a high precision solve otherwise.
        set_attribute(Problem, "BarHomogeneous", 1)
        set_attribute(Problem, "NumericFocus", 3)
        optimize!(Problem)

        # Try to return a suboptimal solution from a possibly early termination as the problem is always feasible and bounded. 
        # (This may be neccesary due to near infeasiblity after convex reformulation.)
        try #is_solved_and_feasible(Problem)
            return objective_value(Problem), value.(order), doubling_count
    
        catch # As a last resort, double the scaled-up ambiguity radius and try again.
            return REMK_intersection_W2_newsvendor_objective_value_and_order(2*ε, demands, weights, doubling_count+1)

        end
    end

end

5
