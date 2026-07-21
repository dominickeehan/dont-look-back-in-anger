using LinearAlgebra
using JuMP, MathOptInterface, Gurobi


# This file expects the following experiment constants to be defined in the
# main script before it is included:
#
# const number_of_items = 3
# const number_of_consumers = 1000


# Each Julia thread reuses its own single-threaded Gurobi environment.
const julia_thread_count =
    Threads.nthreads(:default) + Threads.nthreads(:interactive)
const gurobi_environments =
    Union{Nothing,Gurobi.Env}[nothing for _ in 1:julia_thread_count]
const gurobi_environment_locks =
    [ReentrantLock() for _ in 1:julia_thread_count]


function _gurobi_environment_for_current_thread()
    thread_id = Threads.threadid()
    environment = gurobi_environments[thread_id]
    if isnothing(environment)
        lock(gurobi_environment_locks[thread_id]) do
            if isnothing(gurobi_environments[thread_id])
                gurobi_environments[thread_id] = Gurobi.Env(
                    Dict{String,Any}("OutputFlag" => 0, "Threads" => 1),
                )
            end
            environment = gurobi_environments[thread_id]
        end
    end
    return environment::Gurobi.Env
end


const multi_item_optimizer = optimizer_with_attributes(
    () -> Gurobi.Optimizer(_gurobi_environment_for_current_thread()),
    "BarHomogeneous" => 1,
    "NumericFocus" => 3,
    "FeasibilityTol" => 1.0e-9,
    "BarQCPConvTol" => 1.0e-10,
)


function _new_multi_item_model()
    Problem = Model(multi_item_optimizer)
    set_string_names_on_creation(Problem, false)
    return Problem
end


function _optimize_multi_item_model!(Problem)
    optimize!(Problem)
    is_solved_and_feasible(Problem) && return nothing
    error(
        "Gurobi did not solve the multi-item newsvendor model: " *
        "termination_status=$(termination_status(Problem)), " *
        "primal_status=$(primal_status(Problem))",
    )
end


function _normalized_positive_weights_and_demands(demands, weights)
    positive_weight_indices = weights .> 0.0
    weights = Float64.(weights[positive_weight_indices])
    weights = weights / sum(weights)
    demands = demands[positive_weight_indices]
    return demands, weights
end


# StatsBase's weighted quantile interpolates, but the newsvendor needs the
# discrete inverse empirical CDF.
function _weighted_newsvendor_quantile(values, weights, probability)
    permutation = sortperm(values)
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
# elementary closed form below, minimizing over the order for fixed λ is a
# weighted-quantile problem over per-scenario breakpoints, and the remaining
# one-dimensional function of λ is convex, so a golden-section search solves
# the problem to machine precision.
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


# Evaluates the dual objective at λ, writing the minimizing normalized order
# into order and using the three length-T vectors as workspace.
function _W2_dual_objective!(
    order,
    underage_values,
    overage_values,
    breakpoints,
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
        for t in eachindex(weights)
            demand = normalized_demands[t][i]
            underage_values[t] =
                _bounded_linear_quadratic_conjugate(underage_cost, demand, λ)
            overage_values[t] =
                _bounded_linear_quadratic_conjugate(-overage_cost, demand, λ)
            breakpoints[t] =
                (underage_values[t] - overage_values[t]) /
                (underage_cost + overage_cost)
        end
        order[i] = _weighted_newsvendor_quantile(
            breakpoints,
            weights,
            underage_cost / (underage_cost + overage_cost),
        )
        for t in eachindex(weights)
            objective += weights[t] * max(
                underage_values[t] - underage_cost * order[i],
                overage_values[t] + overage_cost * order[i],
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
    T = length(weights)

    order = zeros(number_of_items)
    underage_values = Vector{Float64}(undef, T)
    overage_values = Vector{Float64}(undef, T)
    breakpoints = Vector{Float64}(undef, T)
    evaluate(λ) = _W2_dual_objective!(
        order,
        underage_values,
        overage_values,
        breakpoints,
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
    while upper - lower > 1.0e-9 * max(1.0, upper)
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
