using LinearAlgebra
using Statistics, StatsBase
using JuMP, MathOptInterface, Gurobi

env = Gurobi.Env()
GRBsetintparam(env, "OutputFlag", 0)
GRBsetintparam(env, "Threads", 1)
optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))


# Construct problem matrices.
# See "Wasserstein Distributionally Robust Optimization with Heterogeneous Data Sources"
# by Yves Rychener, Adrian Esteban-Perez, Juan M. Morales, and Daniel Kuhn (arXiv:2407.13582v2),
# Corollary 2.
loss_pieces = collect(Iterators.product(fill([false, true], number_of_items)...))
a = [[loss_pieces[l][i] ? cu : -co for i in 1:number_of_items] for l in eachindex(loss_pieces)]
b = [-a[l] for l in eachindex(a)]
C = [-Matrix{Float64}(I, number_of_items, number_of_items); Matrix{Float64}(I, number_of_items, number_of_items)]
g = [zeros(number_of_items); number_of_consumers*ones(number_of_items)]


function SO_multi_item_newsvendor_objective_value_and_order(_, demands, weights, doubling_count)

    nonzero_weight_indices = weights .> 0.0
    weights = weights[nonzero_weight_indices]
    weights = weights/sum(weights)
    demands = demands[nonzero_weight_indices]

    T = length(demands)

    Problem = Model(optimizer)

    @variables(Problem, begin
                            number_of_consumers >= order[i=1:number_of_items] >= 0.0
                            s[t=1:T]
                        end)

    #@constraint(Problem, sum(unit_order_costs[i]*order[i] for i in 1:number_of_items) <= order_budget)

    for t in 1:T
        for l in eachindex(a)
            @constraints(Problem, begin
                                    sum(b[l][i]*order[i] for i in 1:number_of_items) + sum(a[l][i]*demands[t][i] for i in 1:number_of_items) <= s[t]
                                  end)
        end
    end

    @objective(Problem, Min, weights'*s)

    set_attribute(Problem, "BarHomogeneous", -1)
    set_attribute(Problem, "NumericFocus", 0)
    optimize!(Problem)

    if is_solved_and_feasible(Problem)
        return objective_value(Problem), value.(order), doubling_count

    else
        set_attribute(Problem, "BarHomogeneous", 1)
        set_attribute(Problem, "NumericFocus", 3)
        optimize!(Problem)

        return objective_value(Problem), value.(order), doubling_count

    end
end


function W2_DRO_multi_item_newsvendor_objective_value_and_order(ε, demands, weights, doubling_count)

    if ε == 0.0; return SO_multi_item_newsvendor_objective_value_and_order(ε, demands, weights, doubling_count); end

    nonzero_weight_indices = weights .> 0.0
    weights = weights[nonzero_weight_indices]
    weights = weights/sum(weights)
    demands = demands[nonzero_weight_indices]

    T = length(demands)

    # See "Wasserstein Distributionally Robust Optimization with Heterogeneous Data Sources"
    # by Yves Rychener, Adrian Esteban-Perez, Juan M. Morales, and Daniel Kuhn (arXiv:2407.13582v2),
    # Corollary 2.
    Problem = Model(optimizer)

    @variables(Problem, begin
                            number_of_consumers >= order[i=1:number_of_items] >= 0.0
                            λ >= 0.0
                            γ[t=1:T]
                            z[t=1:T,l=1:length(a),m=1:length(g)] >= 0.0
                            w[t=1:T,l=1:length(a),i=1:number_of_items]
                        end)

    #@constraint(Problem, sum(unit_order_costs[i]*order[i] for i in 1:number_of_items) <= order_budget)

    for t in 1:T
        for l in eachindex(a)
            @constraints(Problem, begin
                                    # b[l]'*order + w[t,l,:]'*demands[t] + (1/4)*(1/λ)*w[t,l,:]'*w[t,l,:] + z[t,l,:]'*g <= γ[t]
                                    # <==> w[t,l,:]'*w[t,l,:] <= 2*(2*λ)*(γ[t] - b[l]'*order - w[t,l,:]'*demands[t] - z[t,l,:]'*g)
                                    # <==>
                                    [2.0*λ;
                                     γ[t] - sum(b[l][i]*order[i] for i in 1:number_of_items) - sum(w[t,l,i]*demands[t][i] for i in 1:number_of_items) - sum(z[t,l,m]*g[m] for m in 1:length(g));
                                     [w[t,l,i] for i in 1:number_of_items]] in MathOptInterface.RotatedSecondOrderCone(number_of_items + 2)
                                  end)

            for i in 1:number_of_items
                @constraint(Problem, a[l][i] - sum(C[m,i]*z[t,l,m] for m in 1:length(g)) == w[t,l,i])
            end
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

            return W2_DRO_multi_item_newsvendor_objective_value_and_order(2.0*ε, demands, weights, doubling_count+1)

        end
    end
end
