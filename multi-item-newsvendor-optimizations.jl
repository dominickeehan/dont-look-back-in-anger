using JuMP, MathOptInterface, Gurobi


# This file expects the following experiment constants to be defined in the
# main script before it is included:
#
# const number_of_items = 3
# const number_of_consumers = 1000
# const budget = number_of_items * number_of_consumers


# Unit procurement costs are one, so `budget` limits the total order quantity.
const normalized_order_budget = min(budget / number_of_consumers, Float64(number_of_items))
const order_budget_is_not_binding = normalized_order_budget >= number_of_items
const order_budget_feasibility_tolerance = 1.0e-8 * max(1.0, budget)


function _multi_item_order_satisfies_budget(order)
    return order_budget_is_not_binding || sum(order) <= budget + order_budget_feasibility_tolerance
end


# Each Julia thread reuses its own single-threaded Gurobi environment.
const julia_thread_count = Threads.nthreads(:default) + Threads.nthreads(:interactive)
const gurobi_environments = Union{Nothing,Gurobi.Env}[nothing for _ in 1:julia_thread_count]
const gurobi_environment_locks = [ReentrantLock() for _ in 1:julia_thread_count]


function _gurobi_environment_for_current_thread()
    thread_id = Threads.threadid()
    environment = gurobi_environments[thread_id]
    if isnothing(environment)
        lock(gurobi_environment_locks[thread_id]) do
            if isnothing(gurobi_environments[thread_id])
                gurobi_environments[thread_id] = Gurobi.Env(Dict{String,Any}("OutputFlag" => 0, "Threads" => 1))
            end
            environment = gurobi_environments[thread_id]
        end
    end
    return environment::Gurobi.Env
end


const multi_item_optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(_gurobi_environment_for_current_thread()))
const multi_item_first_attempt_barrier_gap_tolerance = 1.0e-6


function _new_multi_item_model()
    problem = Model(multi_item_optimizer)
    set_string_names_on_creation(problem, false) # Stops JuMP from generating solver-facing string names such as order[1].
    return problem
end


function _normalized_positive_weights_and_demands(demands, weights)
    positive_weight_indices = findall(weight -> weight > 0.0, weights)
    normalized_weights = Float64.(weights[positive_weight_indices])
    normalized_weights ./= sum(normalized_weights)
    return demands[positive_weight_indices], normalized_weights
end


function _optimize_multi_item_model!(problem; high_precision = false)
    set_attribute(problem, "BarHomogeneous", -1)
    set_attribute(problem, "NumericFocus", 0)
    if high_precision
        set_attribute(problem, "FeasibilityTol", 1.0e-9)
        set_attribute(problem, "BarQCPConvTol", 1.0e-10)
    else
        set_attribute(problem, "BarQCPConvTol", multi_item_first_attempt_barrier_gap_tolerance)
    end
    optimize!(problem)
    is_solved_and_feasible(problem) && return nothing

    _multi_item_statistics().numeric_retry_solves += 1
    set_attribute(problem, "BarHomogeneous", 1)
    set_attribute(problem, "NumericFocus", 3)
    !high_precision && set_attribute(problem, "BarQCPConvTol", 1.0e-6)
    optimize!(problem)
    is_solved_and_feasible(problem) && return nothing

    error(
        "Gurobi did not solve the multi-item newsvendor model: " *
        "termination_status=$(termination_status(problem)), " *
        "primal_status=$(primal_status(problem))",
    )
end


# StatsBase's weighted quantile interpolates, but the newsvendor needs the discrete inverse empirical CDF.
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


function SO_multi_item_newsvendor_objective_value_and_order(
    _, demands, weights, instance_underage_costs, instance_overage_costs,
)
    demands, weights = _normalized_positive_weights_and_demands(demands, weights)
    critical_fractiles =
        instance_underage_costs ./
        (instance_underage_costs .+ instance_overage_costs)

    order = [
        clamp(
            _weighted_newsvendor_quantile(
                [demands[t][i] for t in eachindex(demands)],
                weights,
                critical_fractiles[i],
            ),
            0.0,
            number_of_consumers,
        )
        for i in 1:number_of_items
    ]

    if _multi_item_order_satisfies_budget(order)
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

    T = length(demands)
    problem = _new_multi_item_model()
    @variables(problem, begin
        number_of_consumers >= order[i = 1:number_of_items] >= 0.0
        underage[t = 1:T, i = 1:number_of_items] >= 0.0
        overage[t = 1:T, i = 1:number_of_items] >= 0.0
    end)
    @constraint(problem, sum(order) <= budget)
    @constraints(problem, begin
        [t = 1:T, i = 1:number_of_items],
            underage[t, i] >= demands[t][i] - order[i]
        [t = 1:T, i = 1:number_of_items],
            overage[t, i] >= order[i] - demands[t][i]
    end)
    @objective(
        problem,
        Min,
        sum(
            weights[t] * (
                instance_underage_costs[i] * underage[t, i] +
                instance_overage_costs[i] * overage[t, i]
            ) for t in 1:T, i in 1:number_of_items
        ),
    )
    _optimize_multi_item_model!(problem)
    return objective_value(problem), value.(order)
end


function _conic_W2_DRO_multi_item_newsvendor_objective_value_and_order(
    epsilon,
    demands,
    weights,
    instance_underage_costs,
    instance_overage_costs,
)
    demands, weights = _normalized_positive_weights_and_demands(demands, weights)
    normalized_demands = [demand ./ number_of_consumers for demand in demands]
    normalized_epsilon = epsilon / number_of_consumers
    T = length(normalized_demands)
    number_of_loss_pieces = 2

    problem = _new_multi_item_model()
    @variables(problem, begin
        1.0 >= normalized_order[i = 1:number_of_items] >= 0.0
        lambda >= 0.0
        gamma[t = 1:T, i = 1:number_of_items]
        z[t = 1:T, i = 1:number_of_items, l = 1:number_of_loss_pieces] >= 0.0
    end)
    if !order_budget_is_not_binding
        @constraint(problem, sum(normalized_order) <= normalized_order_budget)
    end

    # Each loss piece keeps only the support multiplier its worst case can
    # reach: the overage piece has negative xi-slope, so its maximizer
    # xi = d - o / (2 lambda) <= d never leaves xi >= 0, and symmetrically the
    # underage piece only needs xi <= 1. The dropped multipliers are zero at
    # an optimum, so the formulation stays exact.
    for t in 1:T, i in 1:number_of_items, l in 1:number_of_loss_pieces
        demand = normalized_demands[t][i]
        demand_coefficient =
            l == 1 ?
            -instance_overage_costs[i] :
            instance_underage_costs[i]
        order_coefficient = -demand_coefficient
        support_sign = l == 1 ? -1.0 : 1.0
        support_rhs = l == 1 ? 0.0 : 1.0
        @constraint(
            problem,
            [
                2.0 * lambda;
                gamma[t, i] -
                order_coefficient * normalized_order[i] +
                lambda * demand^2 -
                support_rhs * z[t, i, l];
                demand_coefficient +
                2.0 * lambda * demand -
                support_sign * z[t, i, l]
            ] in MathOptInterface.RotatedSecondOrderCone(3),
        )
    end

    @objective(
        problem,
        Min,
        normalized_epsilon^2 * lambda +
        sum(weights[t] * gamma[t, i] for t in 1:T, i in 1:number_of_items),
    )
    _optimize_multi_item_model!(problem)

    objective = number_of_consumers * objective_value(problem)
    order = number_of_consumers .* value.(normalized_order)
    return objective, order
end


struct _WeightedW2DisplacementTerm
    threshold::Float64
    saturated_value::Float64
    inverse_square_coefficient::Float64
end


function _push_weighted_W2_displacement_term!(
    terms, mass, boundary_distance, marginal_cost,
)
    if mass > 0.0 && boundary_distance > 0.0 && marginal_cost > 0.0
        push!(
            terms,
            _WeightedW2DisplacementTerm(
                marginal_cost / (2.0 * boundary_distance),
                mass * boundary_distance^2,
                mass * marginal_cost^2 / 4.0,
            ),
        )
    end
    return nothing
end


# The optional budget multiplier prices ordered units (it is the Lagrange
# multiplier of the order budget). Charging it shifts each item's critical
# fractile from u/(u + o) to (u - mu)/(u + o) while leaving the transport
# displacements' marginal costs at u and o, so the displacement terms below
# remain valid for the budget Lagrangian.
function _prepare_weighted_W2_closed_form(
    demands, weights, instance_underage_costs, instance_overage_costs;
    budget_multiplier = 0.0,
    sort_permutations = nothing,
)
    T = length(demands)
    normalized_demands = Matrix{Float64}(undef, T, number_of_items)
    for t in 1:T, i in 1:number_of_items
        normalized_demands[t, i] = demands[t][i] / number_of_consumers
    end

    quantiles = zeros(number_of_items)
    displacement_terms = _WeightedW2DisplacementTerm[]

    for i in 1:number_of_items
        underage_cost = instance_underage_costs[i]
        overage_cost = instance_overage_costs[i]
        critical_fractile =
            (underage_cost - budget_multiplier) /
            (underage_cost + overage_cost)
        item_demands = view(normalized_demands, :, i)
        permutation = isnothing(sort_permutations) ?
            sortperm(item_demands) : sort_permutations[i]
        quantile_demand = _weighted_newsvendor_quantile(
            item_demands, weights, critical_fractile, permutation,
        )
        quantiles[i] = quantile_demand

        lower_mass = sum(
            (weights[t] for t in 1:T if item_demands[t] < quantile_demand);
            init = 0.0,
        )
        tie_mass = sum(
            (weights[t] for t in 1:T if item_demands[t] == quantile_demand);
            init = 0.0,
        )
        tie_overage_mass = clamp(critical_fractile - lower_mass, 0.0, tie_mass)
        tie_underage_mass = tie_mass - tie_overage_mass

        for t in 1:T
            demand = item_demands[t]
            if demand < quantile_demand
                _push_weighted_W2_displacement_term!(
                    displacement_terms, weights[t], demand, overage_cost,
                )
            elseif demand > quantile_demand
                _push_weighted_W2_displacement_term!(
                    displacement_terms, weights[t], 1.0 - demand, underage_cost,
                )
            end
        end

        _push_weighted_W2_displacement_term!(
            displacement_terms, tie_overage_mass, quantile_demand, overage_cost,
        )
        _push_weighted_W2_displacement_term!(
            displacement_terms,
            tie_underage_mass,
            1.0 - quantile_demand,
            underage_cost,
        )
    end

    sort!(displacement_terms, by = term -> term.threshold)
    return normalized_demands, quantiles, displacement_terms
end


function _optimal_weighted_W2_lambda(
    normalized_epsilon_squared, displacement_terms,
)
    saturated_sum = sum(
        (term.saturated_value for term in displacement_terms);
        init = 0.0,
    )
    normalized_epsilon_squared >= saturated_sum && return 0.0

    inverse_square_sum = 0.0
    term_index = 1
    while term_index <= length(displacement_terms)
        interval_lower = displacement_terms[term_index].threshold
        while term_index <= length(displacement_terms) &&
              displacement_terms[term_index].threshold == interval_lower
            saturated_sum -= displacement_terms[term_index].saturated_value
            inverse_square_sum +=
                displacement_terms[term_index].inverse_square_coefficient
            term_index += 1
        end
        interval_upper = term_index <= length(displacement_terms) ?
                         displacement_terms[term_index].threshold : Inf
        denominator = normalized_epsilon_squared - saturated_sum
        if denominator > 0.0
            candidate = sqrt(inverse_square_sum / denominator)
            tolerance = 128.0 * eps(Float64) * max(1.0, interval_lower)
            if interval_lower - tolerance <= candidate <= interval_upper + tolerance
                return clamp(candidate, interval_lower, interval_upper)
            end
        end
    end
    error("failed to identify the weighted-W2 multiplier interval")
end


function _bounded_linear_quadratic_conjugate(slope, demand, lambda)
    if iszero(lambda)
        return max(0.0, slope)
    elseif slope >= 0.0
        displacement = min(1.0 - demand, slope / (2.0 * lambda))
        return slope * demand + slope * displacement - lambda * displacement^2
    else
        displacement = min(demand, -slope / (2.0 * lambda))
        return slope * demand - slope * displacement - lambda * displacement^2
    end
end


function _weighted_W2_normalized_objective(
    normalized_epsilon_squared,
    lambda,
    normalized_order,
    normalized_demands,
    weights,
    instance_underage_costs,
    instance_overage_costs,
)
    normalized_objective = normalized_epsilon_squared * lambda
    for t in eachindex(weights), i in 1:number_of_items
        demand = normalized_demands[t, i]
        underage_cost = instance_underage_costs[i]
        overage_cost = instance_overage_costs[i]
        normalized_objective += weights[t] * max(
            -underage_cost * normalized_order[i] +
            _bounded_linear_quadratic_conjugate(
                underage_cost, demand, lambda,
            ),
            overage_cost * normalized_order[i] +
            _bounded_linear_quadratic_conjugate(
                -overage_cost, demand, lambda,
            ),
        )
    end
    return normalized_objective
end


function _solve_weighted_W2_closed_form(
    epsilon,
    normalized_demands,
    quantiles,
    displacement_terms,
    weights,
    instance_underage_costs,
    instance_overage_costs,
)
    normalized_epsilon_squared = (epsilon / number_of_consumers)^2
    lambda = _optimal_weighted_W2_lambda(
        normalized_epsilon_squared, displacement_terms,
    )

    normalized_order = zeros(number_of_items)
    if iszero(lambda)
        normalized_order .=
            instance_underage_costs ./
            (instance_underage_costs .+ instance_overage_costs)
    else
        for i in 1:number_of_items
            underage_cost = instance_underage_costs[i]
            overage_cost = instance_overage_costs[i]
            normalized_order[i] = clamp(
                (
                    _bounded_linear_quadratic_conjugate(
                        underage_cost, quantiles[i], lambda,
                    ) -
                    _bounded_linear_quadratic_conjugate(
                        -overage_cost, quantiles[i], lambda,
                    )
                ) /
                (underage_cost + overage_cost),
                0.0,
                1.0,
            )
        end
    end

    normalized_objective = _weighted_W2_normalized_objective(
        normalized_epsilon_squared,
        lambda,
        normalized_order,
        normalized_demands,
        weights,
        instance_underage_costs,
        instance_overage_costs,
    )

    objective = number_of_consumers * normalized_objective
    order = number_of_consumers .* normalized_order
    return objective, order
end


const multi_item_budget_dual_bisection_iterations = 60
const multi_item_budget_dual_multiplier_tolerance = 1.0e-12
const multi_item_budget_dual_relative_gap_tolerance = 1.0e-8
const multi_item_budget_dual_absolute_gap_tolerance = 1.0e-10


# Minimizes the weighted-W2 Lagrangian at a fixed budget multiplier: the
# unconstrained closed form at the shifted critical fractile. An item whose
# underage cost is not above the multiplier orders nothing.
function _weighted_W2_lagrangian_lambda_and_order(
    normalized_epsilon_squared,
    demands,
    weights,
    instance_underage_costs,
    instance_overage_costs,
    budget_multiplier,
    sort_permutations,
)
    _, quantiles, displacement_terms = _prepare_weighted_W2_closed_form(
        demands, weights, instance_underage_costs, instance_overage_costs;
        budget_multiplier,
        sort_permutations,
    )
    lambda = _optimal_weighted_W2_lambda(
        normalized_epsilon_squared, displacement_terms,
    )
    normalized_order = zeros(number_of_items)
    for i in 1:number_of_items
        underage_cost = instance_underage_costs[i]
        overage_cost = instance_overage_costs[i]
        budget_multiplier >= underage_cost && continue
        normalized_order[i] = clamp(
            (
                _bounded_linear_quadratic_conjugate(
                    underage_cost, quantiles[i], lambda,
                ) -
                _bounded_linear_quadratic_conjugate(
                    -overage_cost, quantiles[i], lambda,
                )
            ) / (underage_cost + overage_cost),
            0.0,
            1.0,
        )
    end
    return lambda, normalized_order
end


# Solves the budget-constrained weighted-W2 problem by dualizing the budget,
# whose multiplier is the only coupling between items. The total Lagrangian
# order is nonincreasing in the multiplier, so bisection brackets the value at
# which the budget binds; the primal candidate interpolates between the final
# bracket's orders because the optimal order can lie between fractile jumps.
# A duality-gap certificate guards the result, returning nothing to request
# the conic fallback.
function _budget_dual_weighted_W2_solution(
    epsilon,
    demands,
    weights,
    instance_underage_costs,
    instance_overage_costs,
)
    normalized_epsilon_squared = (epsilon / number_of_consumers)^2
    T = length(demands)
    normalized_demands = Matrix{Float64}(undef, T, number_of_items)
    for t in 1:T, i in 1:number_of_items
        normalized_demands[t, i] = demands[t][i] / number_of_consumers
    end
    # The demand sort order does not depend on the budget multiplier, so sort
    # once here instead of at every quantile evaluation in the bisection.
    sort_permutations = [
        sortperm(view(normalized_demands, :, i)) for i in 1:number_of_items
    ]
    lagrangian_lambda_and_order(budget_multiplier) =
        _weighted_W2_lagrangian_lambda_and_order(
            normalized_epsilon_squared,
            demands,
            weights,
            instance_underage_costs,
            instance_overage_costs,
            budget_multiplier,
            sort_permutations,
        )
    objective_at(lambda, normalized_order) = _weighted_W2_normalized_objective(
        normalized_epsilon_squared,
        lambda,
        normalized_order,
        normalized_demands,
        weights,
        instance_underage_costs,
        instance_overage_costs,
    )

    lower_multiplier = 0.0
    lower_lambda, lower_order = lagrangian_lambda_and_order(lower_multiplier)
    upper_multiplier = maximum(instance_underage_costs)
    upper_lambda, upper_order = lagrangian_lambda_and_order(upper_multiplier)
    multiplier_tolerance =
        multi_item_budget_dual_multiplier_tolerance * upper_multiplier
    for _ in 1:multi_item_budget_dual_bisection_iterations
        upper_multiplier - lower_multiplier <= multiplier_tolerance && break
        multiplier = 0.5 * (lower_multiplier + upper_multiplier)
        lambda, normalized_order = lagrangian_lambda_and_order(multiplier)
        if sum(normalized_order) > normalized_order_budget
            lower_multiplier, lower_lambda, lower_order =
                multiplier, lambda, normalized_order
        else
            upper_multiplier, upper_lambda, upper_order =
                multiplier, lambda, normalized_order
        end
    end

    lower_total = sum(lower_order)
    upper_total = sum(upper_order)
    interpolation = lower_total > upper_total ?
        clamp(
            (normalized_order_budget - upper_total) /
            (lower_total - upper_total),
            0.0,
            1.0,
        ) :
        0.0
    normalized_order =
        interpolation .* lower_order .+ (1.0 - interpolation) .* upper_order

    # The Lagrangian objective is jointly convex in (lambda, order), so the
    # interpolated pair is itself Lagrangian-optimal at the bracketed
    # multiplier and evaluating the primal candidate there closes the duality
    # gap; the endpoint lambdas would overestimate it and force the conic
    # fallback below.
    interpolated_lambda =
        interpolation * lower_lambda + (1.0 - interpolation) * upper_lambda
    primal_value = objective_at(interpolated_lambda, normalized_order)
    dual_value = max(
        objective_at(lower_lambda, lower_order) +
            lower_multiplier * (lower_total - normalized_order_budget),
        objective_at(upper_lambda, upper_order) +
            upper_multiplier * (upper_total - normalized_order_budget),
    )
    gap_tolerance =
        multi_item_budget_dual_absolute_gap_tolerance +
        multi_item_budget_dual_relative_gap_tolerance * abs(primal_value)
    if primal_value - dual_value > gap_tolerance
        _multi_item_statistics().budget_dual_failures += 1
        return nothing
    end

    _multi_item_statistics().budget_dual_solutions += 1
    return (
        number_of_consumers * primal_value,
        number_of_consumers .* normalized_order,
    )
end


function W2_DRO_multi_item_newsvendor_objective_value_and_order(
    epsilon,
    demands,
    weights,
    instance_underage_costs,
    instance_overage_costs,
)
    if iszero(epsilon)
        return SO_multi_item_newsvendor_objective_value_and_order(
            epsilon,
            demands,
            weights,
            instance_underage_costs,
            instance_overage_costs,
        )
    end

    demands, weights = _normalized_positive_weights_and_demands(demands, weights)
    normalized_demands, quantiles, displacement_terms =
        _prepare_weighted_W2_closed_form(
            demands,
            weights,
            instance_underage_costs,
            instance_overage_costs,
        )
    closed_form_solution = _solve_weighted_W2_closed_form(
        epsilon,
        normalized_demands,
        quantiles,
        displacement_terms,
        weights,
        instance_underage_costs,
        instance_overage_costs,
    )
    _multi_item_order_satisfies_budget(closed_form_solution[2]) &&
        return closed_form_solution
    budget_dual_solution = _budget_dual_weighted_W2_solution(
        epsilon,
        demands,
        weights,
        instance_underage_costs,
        instance_overage_costs,
    )
    isnothing(budget_dual_solution) || return budget_dual_solution
    return _conic_W2_DRO_multi_item_newsvendor_objective_value_and_order(
        epsilon,
        demands,
        weights,
        instance_underage_costs,
        instance_overage_costs,
    )
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
    for weight_index in eachindex(weight_vectors), radius_index in eachindex(ambiguity_radii)
        results[radius_index, weight_index] = newsvendor_objective_value_and_order(
            ambiguity_radii[radius_index],
            demands,
            weight_vectors[weight_index],
            instance_underage_costs,
            instance_overage_costs,
        )
    end
    return results
end


function _multi_item_newsvendor_grid(
    ::typeof(W2_DRO_multi_item_newsvendor_objective_value_and_order),
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
        active_demands, normalized_weights =
            _normalized_positive_weights_and_demands(
                demands, weight_vectors[weight_index],
            )
        closed_form_data = _prepare_weighted_W2_closed_form(
            active_demands,
            normalized_weights,
            instance_underage_costs,
            instance_overage_costs,
        )

        for radius_index in eachindex(ambiguity_radii)
            epsilon = ambiguity_radii[radius_index]
            if iszero(epsilon)
                results[radius_index, weight_index] =
                    SO_multi_item_newsvendor_objective_value_and_order(
                        epsilon,
                        active_demands,
                        normalized_weights,
                        instance_underage_costs,
                        instance_overage_costs,
                    )
            else
                normalized_demands, quantiles, displacement_terms = closed_form_data
                closed_form_solution = _solve_weighted_W2_closed_form(
                    epsilon,
                    normalized_demands,
                    quantiles,
                    displacement_terms,
                    normalized_weights,
                    instance_underage_costs,
                    instance_overage_costs,
                )
                results[radius_index, weight_index] =
                    if _multi_item_order_satisfies_budget(closed_form_solution[2])
                        closed_form_solution
                    else
                        budget_dual_solution = _budget_dual_weighted_W2_solution(
                            epsilon,
                            active_demands,
                            normalized_weights,
                            instance_underage_costs,
                            instance_overage_costs,
                        )
                        isnothing(budget_dual_solution) ?
                        _conic_W2_DRO_multi_item_newsvendor_objective_value_and_order(
                            epsilon,
                            active_demands,
                            normalized_weights,
                            instance_underage_costs,
                            instance_overage_costs,
                        ) :
                        budget_dual_solution
                    end
            end
        end
    end
    return results
end


include("multi-item-newsvendor-intersection-optimizations.jl")
