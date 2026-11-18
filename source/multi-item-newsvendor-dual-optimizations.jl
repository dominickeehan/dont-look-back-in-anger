# Fast solvers for the newsvendor problems of multi-item-newsvendor-conic-optimizations.jl that do
# not need a conic solver. The experiments use these (see test-dual-optimizations.jl for a comparison).
#
# Included after weights.jl by scripts that define number_of_items and number_of_consumers.


# Samples with positive weight and their renormalized weights.
function _normalized_positive_weights_and_demands(demands, weights)

        positive_weight_indices = weights .> 0.0
        weights = Float64.(weights[positive_weight_indices])
        weights = weights / sum(weights)
        demands = demands[positive_weight_indices]

        return demands, weights
end


# Smallest value whose cumulative weight is at least the probability (StatsBase's weighted quantile
# interpolates instead).
function _weighted_newsvendor_quantile(values, weights, probability, permutation = sortperm(values))

        cumulative_weight = 0.0
        for position in eachindex(permutation)
                index = permutation[position]
                cumulative_weight += weights[index]
                (cumulative_weight >= probability || position == lastindex(permutation)) && return values[index]

        end
end


# Weighted sample average approximation. The order for each item is a weighted quantile of its demands.
function SO_multi_item_newsvendor_objective_value_and_order(_, demands, weights, instance_underage_costs, instance_overage_costs)

        demands, weights = _normalized_positive_weights_and_demands(demands, weights)

        order = [_weighted_newsvendor_quantile([demands[t][i] for t in eachindex(demands)], weights, instance_underage_costs[i] / (instance_underage_costs[i] + instance_overage_costs[i])) for i in 1:number_of_items]

        objective = sum(weights[t] * sum(instance_underage_costs[i] * max(demands[t][i] - order[i], 0.0) + instance_overage_costs[i] * max(order[i] - demands[t][i], 0.0) for i in 1:number_of_items) for t in eachindex(demands))

        return objective, order
end


# Supremum of slope*ξ - λ*(ξ - demand)^2 over ξ in [0,1]. It is attained by moving |slope|/(2λ) from
# the demand in the direction of the slope or to the end of the interval if that is nearer.
function _bounded_linear_quadratic_conjugate(slope, demand, λ)

        if iszero(λ)
                return max(0.0, slope)
        elseif slope >= 0.0
                displacement = min(1.0 - demand, slope / (2.0 * λ))
                return slope * (demand + displacement) - λ * displacement^2
        else
                displacement = min(demand, -slope / (2.0 * λ))
                return slope * (demand - displacement) - λ * displacement^2
        end
end


# Dual objective value of the weighted 2-Wasserstein DRO newsvendor problem at λ.
#
# Corollary 2 of Rychener et al. gives (for demands scaled to [0,1])
#
#       min λ*ε^2 + Σ_t weights[t] Σ_i max(A[t,i] - u[i]*order[i], B[t,i] + o[i]*order[i])
#
# over λ ≥ 0 and the order. Here u and o are the underage and overage costs. A[t,i] is the supremum
# of u[i]*ξ - λ*(ξ - demands[t][i])^2 over ξ in [0,1] and B[t,i] is that of -o[i]*ξ - λ*(ξ - demands[t][i])^2.
#
# For a fixed λ this is a weighted newsvendor problem in the breakpoints
# (A[t,i] - B[t,i])/(u[i] + o[i]) so order[i] is their weighted quantile. As A[t,i] is nondecreasing
# and B[t,i] is nonincreasing in demands[t][i] this is the breakpoint of the weighted quantile of
# the demands. The maximum is then the overage piece for demands below this quantile and the
# underage piece for the others.
function _W2_dual_objective!(order, quantile_demands, λ, normalized_demands, weights, instance_underage_costs, instance_overage_costs, normalized_epsilon)

        objective = λ * normalized_epsilon^2
        for i in 1:number_of_items

                underage_cost = instance_underage_costs[i]
                overage_cost = instance_overage_costs[i]
                quantile_demand = quantile_demands[i]
                underage_value = _bounded_linear_quadratic_conjugate(underage_cost, quantile_demand, λ)
                overage_value = _bounded_linear_quadratic_conjugate(-overage_cost, quantile_demand, λ)
                order[i] = (underage_value - overage_value) / (underage_cost + overage_cost)

                for t in eachindex(weights)
                        demand = normalized_demands[t][i]
                        slope = demand < quantile_demand ? -overage_cost : underage_cost
                        objective += weights[t] * (_bounded_linear_quadratic_conjugate(slope, demand, λ) - slope * order[i])

                end
        end

        return objective
end


# 2-Wasserstein DRO around the weighted empirical distribution of the demands.
function W2_DRO_multi_item_newsvendor_objective_value_and_order(ε, demands, weights, instance_underage_costs, instance_overage_costs)

        # Sample average approximation for a radius of zero.
        if ε == 0.0; return SO_multi_item_newsvendor_objective_value_and_order(ε, demands, weights, instance_underage_costs, instance_overage_costs); end

        demands, weights = _normalized_positive_weights_and_demands(demands, weights)

        # Demands and radius divided by the number of consumers to keep the problem numerically stable
        # (undone on return).
        normalized_demands = [demand ./ number_of_consumers for demand in demands]
        normalized_epsilon = ε / number_of_consumers

        order = zeros(number_of_items)

        # Weighted quantiles of the demands (the same for every λ) and the dual objective at λ.
        quantile_demands = [_weighted_newsvendor_quantile([demand[i] for demand in normalized_demands], weights, instance_underage_costs[i] / (instance_underage_costs[i] + instance_overage_costs[i])) for i in 1:number_of_items]
        evaluate(λ) = _W2_dual_objective!(order, quantile_demands, λ, normalized_demands, weights, instance_underage_costs, instance_overage_costs, normalized_epsilon)

        # Bracket [0, upper] for λ. For λ beyond upper the worst-case demands are within ε of the samples so
        # the objective is nondecreasing there.
        lower = 0.0
        upper = sqrt(sum(max(instance_underage_costs[i], instance_overage_costs[i])^2 for i in 1:number_of_items)) / (2.0 * normalized_epsilon)

        # Golden-section search over λ for the minimum of the convex dual objective.
        golden_ratio_fraction = (sqrt(5.0) - 1.0) / 2.0
        first_λ = upper - golden_ratio_fraction * (upper - lower)
        second_λ = lower + golden_ratio_fraction * (upper - lower)
        first_objective = evaluate(first_λ)
        second_objective = evaluate(second_λ)
        while upper - lower > 1.0e-6 * max(1.0, upper)
                if first_objective <= second_objective
                        upper, second_λ, second_objective = second_λ, first_λ, first_objective
                        first_λ = upper - golden_ratio_fraction * (upper - lower)
                        first_objective = evaluate(first_λ)
                else
                        lower, first_λ, first_objective = first_λ, second_λ, second_objective
                        second_λ = lower + golden_ratio_fraction * (upper - lower)
                        second_objective = evaluate(second_λ)
                end

        end
        objective = evaluate((lower + upper) / 2.0)

        return number_of_consumers * objective, number_of_consumers .* order
end


# Objective values and orders for every ambiguity radius and weight vector. The intersection
# approach has its own version that computes the demand distances only once.
function _multi_item_newsvendor_grid(newsvendor_objective_value_and_order, ambiguity_radii, demands, weight_vectors, instance_underage_costs, instance_overage_costs)

        result_type = Tuple{Float64,Vector{Float64}}
        results = Matrix{result_type}(undef, length(ambiguity_radii), length(weight_vectors))
        for weight_index in eachindex(weight_vectors)
                for radius_index in eachindex(ambiguity_radii)
                        results[radius_index, weight_index] = newsvendor_objective_value_and_order(ambiguity_radii[radius_index], demands, weight_vectors[weight_index], instance_underage_costs, instance_overage_costs)

                end
        end

        return results
end


include("multi-item-newsvendor-intersection-dual-optimizations.jl")
