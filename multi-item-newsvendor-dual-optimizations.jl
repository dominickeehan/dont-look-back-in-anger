# This file expects the following experiment constants to be defined in the
# main script before it is included:
#
# const number_of_items = 3
# const number_of_consumers = 1000


function _normalized_positive_weights_and_demands(demands, weights)
    positive_weight_indices = weights .> 0.0
    weights = Float64.(weights[positive_weight_indices])
    weights = weights / sum(weights)
    demands = demands[positive_weight_indices]
    return demands, weights
end


# StatsBase's weighted quantile interpolates, but the newsvendor needs the
# discrete inverse empirical CDF.
function _weighted_newsvendor_quantile(
    values, weights, probability, permutation = sortperm(values),
)
    cumulative_weight = 0.0
    for position in eachindex(permutation)
        index = permutation[position]
        cumulative_weight += weights[index]
        (cumulative_weight >= probability || position == lastindex(permutation)) &&
            return values[index]
    end
end


# The newsvendor loss separates across items, so the weighted sample-average
# problem is solved by per-item weighted quantiles at the critical fractiles.
function SO_multi_item_newsvendor_objective_value_and_order(
    _,
    demands,
    weights,
    instance_underage_costs,
    instance_overage_costs,
)
    demands, weights =
        _normalized_positive_weights_and_demands(demands, weights)

    order = [
        _weighted_newsvendor_quantile(
            [demands[t][i] for t in eachindex(demands)],
            weights,
            instance_underage_costs[i] /
            (instance_underage_costs[i] + instance_overage_costs[i]),
        )
        for i in 1:number_of_items
    ]

    objective = sum(
        weights[t] * sum(
            instance_underage_costs[i] *
            max(demands[t][i] - order[i], 0.0) +
            instance_overage_costs[i] *
            max(order[i] - demands[t][i], 0.0)
            for i in 1:number_of_items
        ) for t in eachindex(demands)
    )
    return objective, order
end


# Weighted type-2 Wasserstein DRO over the box support. By strong duality
# (Corollary 2 of "Wasserstein Distributionally Robust Optimization with
# Heterogeneous Data Sources" by Rychener, Esteban-Perez, Morales, and Kuhn)
# the problem, stated on the normalized support [0,1]^number_of_items, equals
#
#   min_{λ ≥ 0, order} λ ε² + Σ_t weights[t] Σ_i sup_{ξ ∈ [0,1]}
#     [max(uᵢ (ξ - orderᵢ), oᵢ (orderᵢ - ξ)) - λ (ξ - demands[t][i])²],
#
# where the supremum decomposes across items because the loss and the squared
# Euclidean transport cost both separate coordinate-wise. The supremum has the
# elementary closed form below. The per-scenario breakpoints are nondecreasing
# in demand for every λ, so the minimizing order is always determined by the
# same weighted-quantile position. The remaining one-dimensional function of λ
# is convex, so a golden-section search solves the problem to the specified
# numerical tolerance.
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


# Evaluates the dual objective at λ and writes the minimizing normalized order.
# Demands below the fixed quantile use the overage piece; demands at or above it
# use the underage piece. Either piece is valid when their breakpoints tie.
function _W2_dual_objective!(
    order,
    quantile_demands,
    λ,
    normalized_demands,
    weights,
    instance_underage_costs,
    instance_overage_costs,
    normalized_epsilon,
)
    objective = λ * normalized_epsilon^2
    for i in 1:number_of_items
        underage_cost = instance_underage_costs[i]
        overage_cost = instance_overage_costs[i]
        quantile_demand = quantile_demands[i]
        underage_value = _bounded_linear_quadratic_conjugate(
            underage_cost, quantile_demand, λ,
        )
        overage_value = _bounded_linear_quadratic_conjugate(
            -overage_cost, quantile_demand, λ,
        )
        order[i] = (underage_value - overage_value) /
            (underage_cost + overage_cost)

        for t in eachindex(weights)
            demand = normalized_demands[t][i]
            slope = demand < quantile_demand ? -overage_cost : underage_cost
            objective += weights[t] * (
                _bounded_linear_quadratic_conjugate(
                    slope, demand, λ,
                ) - slope * order[i]
            )
        end
    end
    return objective
end


function W2_DRO_multi_item_newsvendor_objective_value_and_order(
    ε,
    demands,
    weights,
    instance_underage_costs,
    instance_overage_costs,
)
    if ε == 0.0
        return SO_multi_item_newsvendor_objective_value_and_order(
            ε,
            demands,
            weights,
            instance_underage_costs,
            instance_overage_costs,
        )
    end

    demands, weights =
        _normalized_positive_weights_and_demands(demands, weights)
    normalized_demands = [demand ./ number_of_consumers for demand in demands]
    normalized_epsilon = ε / number_of_consumers

    order = zeros(number_of_items)
    quantile_demands = [
        _weighted_newsvendor_quantile(
            [demand[i] for demand in normalized_demands],
            weights,
            instance_underage_costs[i] /
                (instance_underage_costs[i] + instance_overage_costs[i]),
        )
        for i in 1:number_of_items
    ]
    evaluate(λ) = _W2_dual_objective!(
        order,
        quantile_demands,
        λ,
        normalized_demands,
        weights,
        instance_underage_costs,
        instance_overage_costs,
        normalized_epsilon,
    )

    # Every worst-case displacement is at most cost / (2λ) per item, so above
    # this multiplier the dual derivative ε² - Σ_t weights[t] ‖ξ_t - ξ̂_t‖² is
    # nonnegative and the minimizer lies within the bracket.
    lower = 0.0
    upper = sqrt(sum(
        max(instance_underage_costs[i], instance_overage_costs[i])^2
        for i in 1:number_of_items
    )) / (2.0 * normalized_epsilon)

    golden_ratio_fraction = (sqrt(5.0) - 1.0) / 2.0
    first_λ = upper - golden_ratio_fraction * (upper - lower)
    second_λ = lower + golden_ratio_fraction * (upper - lower)
    first_objective = evaluate(first_λ)
    second_objective = evaluate(second_λ)
    while upper - lower > 1.0e-6 * max(1.0, upper)
        if first_objective <= second_objective
            upper, second_λ, second_objective =
                second_λ, first_λ, first_objective
            first_λ = upper - golden_ratio_fraction * (upper - lower)
            first_objective = evaluate(first_λ)
        else
            lower, first_λ, first_objective =
                first_λ, second_λ, second_objective
            second_λ = lower + golden_ratio_fraction * (upper - lower)
            second_objective = evaluate(second_λ)
        end
    end
    objective = evaluate((lower + upper) / 2.0)

    return number_of_consumers * objective, number_of_consumers .* order
end


function _multi_item_newsvendor_grid(
    newsvendor_objective_value_and_order,
    ambiguity_radii,
    demands,
    weight_vectors,
    instance_underage_costs,
    instance_overage_costs,
)
    result_type = Tuple{Float64,Vector{Float64}}
    results = Matrix{result_type}(
        undef, length(ambiguity_radii), length(weight_vectors),
    )
    for weight_index in eachindex(weight_vectors)
        for radius_index in eachindex(ambiguity_radii)
            results[radius_index, weight_index] =
                newsvendor_objective_value_and_order(
                    ambiguity_radii[radius_index],
                    demands,
                    weight_vectors[weight_index],
                    instance_underage_costs,
                    instance_overage_costs,
                )
        end
    end
    return results
end


include("multi-item-newsvendor-intersection-dual-optimizations.jl")
