using JuMP, MathOptInterface, Gurobi


# This file expects the following experiment constants to be defined in the
# main script before it is included:
#
# const number_of_items = 3
# const number_of_consumers = 1000


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


function _prepare_weighted_W2_closed_form(
    demands, weights, instance_underage_costs, instance_overage_costs,
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
        critical_fractile = underage_cost / (underage_cost + overage_cost)
        item_demands = view(normalized_demands, :, i)
        quantile_demand = _weighted_newsvendor_quantile(
            item_demands, weights, critical_fractile,
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
    return _solve_weighted_W2_closed_form(
        epsilon,
        normalized_demands,
        quantiles,
        displacement_terms,
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
                results[radius_index, weight_index] =
                    _solve_weighted_W2_closed_form(
                        epsilon,
                        normalized_demands,
                        quantiles,
                        displacement_terms,
                        normalized_weights,
                        instance_underage_costs,
                        instance_overage_costs,
                    )
            end
        end
    end
    return results
end


include("multi-item-newsvendor-intersection-optimizations.jl")
