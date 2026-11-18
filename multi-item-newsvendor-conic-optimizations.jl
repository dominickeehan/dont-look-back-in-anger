using LinearAlgebra
using JuMP, MathOptInterface, Gurobi

# Newsvendor problems as conic programs. The experiments use the faster solvers in multi-item-newsvendor-dual-optimizations.jl instead.
#
# Included after weights.jl by scripts that define number_of_items and number_of_consumers.

# Gurobi optimizer with no terminal output and one thread.
env = Gurobi.Env()
GRBsetintparam(env, "OutputFlag", 0)
GRBsetintparam(env, "Threads", 1)
optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))


# Samples with positive weight and their renormalized weights.
function positive_weight_samples(demands, weights)

        positive = weights .> 0
        weights = weights[positive]

        return demands[positive], weights/sum(weights)
end

# Problem matrices from Corollary 2 of "Wasserstein Distributionally Robust Optimization with
# Heterogeneous Data Sources" by Rychener et al. (arXiv:2407.13582v2) for demands scaled to [0,1].
function problem_matrices(underage_costs, overage_costs)

        loss_pieces = collect(Iterators.product(fill([false, true], number_of_items)...))
        a = [[loss_pieces[l][i] ? underage_costs[i] : -overage_costs[i] for i in 1:number_of_items] for l in eachindex(loss_pieces)]
        b = [-a[l] for l in eachindex(a)]
        C = [-Matrix{Float64}(I, number_of_items, number_of_items); Matrix{Float64}(I, number_of_items, number_of_items)]
        g = [zeros(number_of_items); ones(number_of_items)]

        return a, b, C, g
end

function optimize_with_high_precision_fallback!(Problem)

        optimize!(Problem)
        if is_solved_and_feasible(Problem); return; end

        # Otherwise retry at high precision.
        set_attribute(Problem, "BarHomogeneous", 1)
        set_attribute(Problem, "NumericFocus", 3)
        optimize!(Problem)

        if !is_solved_and_feasible(Problem); error("Gurobi did not solve the problem: $(termination_status(Problem))."); end
end


# Weighted sample average approximation as a linear program.
function SO_multi_item_newsvendor_objective_value_and_order(_, demands, weights, underage_costs, overage_costs)

        demands, weights = positive_weight_samples(demands, weights)
        a, b, _, _ = problem_matrices(underage_costs, overage_costs)

        T = length(demands)

        Problem = Model(optimizer)

        @variables(Problem, begin
                                number_of_consumers >= order[i=1:number_of_items] >= 0.0
                                s[t=1:T]
                            end)

        for t in 1:T, l in eachindex(a)
                @constraint(Problem, b[l]'*order + a[l]'*demands[t] <= s[t])

        end

        @objective(Problem, Min, weights'*s)

        optimize_with_high_precision_fallback!(Problem)

        return objective_value(Problem), value.(order)
end


# 2-Wasserstein DRO around the weighted empirical distribution of the demands.
function W2_DRO_multi_item_newsvendor_objective_value_and_order(ε, demands, weights, underage_costs, overage_costs)

        if ε == 0.0; return SO_multi_item_newsvendor_objective_value_and_order(ε, demands, weights, underage_costs, overage_costs); end

        demands, weights = positive_weight_samples(demands, weights)
        a, b, C, g = problem_matrices(underage_costs, overage_costs)

        # Demands and radius divided by the number of consumers to keep the problem numerically stable
        # (undone on return).
        T = length(demands)
        d = [demand/number_of_consumers for demand in demands]
        ε = ε/number_of_consumers

        Problem = Model(optimizer)

        @variables(Problem, begin
                                1.0 >= order[i=1:number_of_items] >= 0.0
                                λ >= 0.0
                                γ[t=1:T]
                                z[t=1:T,l=1:length(a),m=1:length(g)] >= 0.0
                                w[t=1:T,l=1:length(a),i=1:number_of_items]
                            end)

        for t in 1:T, l in eachindex(a)
                @constraints(Problem, begin
                                        # b[l]'*order + w[t,l,:]'*d[t] + (1/4)*(1/λ)*w[t,l,:]'*w[t,l,:] + z[t,l,:]'*g <= γ[t]
                                        # <==> w[t,l,:]'*w[t,l,:] <= 2*(2*λ)*(γ[t] - b[l]'*order - w[t,l,:]'*d[t] - z[t,l,:]'*g)
                                        # <==>
                                        [2.0*λ; γ[t] - b[l]'*order - w[t,l,:]'*d[t] - z[t,l,:]'*g; w[t,l,:]] in MathOptInterface.RotatedSecondOrderCone(number_of_items + 2)

                                        a[l] - C'*z[t,l,:] .== w[t,l,:]
                                      end)

        end

        @objective(Problem, Min, (ε^2)*λ + weights'*γ)

        optimize_with_high_precision_fallback!(Problem)

        return number_of_consumers*objective_value(Problem), number_of_consumers*value.(order)
end


include("multi-item-newsvendor-intersection-conic-optimizations.jl")
