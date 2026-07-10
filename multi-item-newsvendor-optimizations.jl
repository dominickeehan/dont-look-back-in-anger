using JuMP, MathOptInterface, Gurobi


const multi_item_dimension = Int(number_of_items)
const multi_item_demand_upper_bound = Float64(number_of_consumers)
const multi_item_underage_cost = Float64(cu)
const multi_item_overage_cost = Float64(co)
multi_item_dimension > 0 || throw(ArgumentError("number_of_items must be positive"))
isfinite(multi_item_demand_upper_bound) && multi_item_demand_upper_bound > 0.0 ||
    throw(ArgumentError("number_of_consumers must be finite and positive"))
isfinite(multi_item_underage_cost) && isfinite(multi_item_overage_cost) ||
    throw(ArgumentError("cu and co must be finite"))
multi_item_underage_cost >= 0.0 && multi_item_overage_cost >= 0.0 ||
    throw(ArgumentError("cu and co must be nonnegative"))
multi_item_underage_cost + multi_item_overage_cost > 0.0 ||
    throw(ArgumentError("at least one of cu and co must be positive"))


# A Gurobi environment must not be shared by models running on different Julia
# threads. The plotting script uses static thread scheduling, so one environment
# per Julia thread is both safe and cheaper than constructing one per model.
const multi_item_julia_thread_count =
    Threads.nthreads(:default) + Threads.nthreads(:interactive)
const multi_item_gurobi_environments = Union{Nothing,Gurobi.Env}[
    nothing for _ in 1:multi_item_julia_thread_count
]
const multi_item_gurobi_environment_locks = [
    ReentrantLock() for _ in 1:multi_item_julia_thread_count
]


function _multi_item_gurobi_environment_for_current_thread()
    thread_id = Threads.threadid()
    environment = multi_item_gurobi_environments[thread_id]
    if isnothing(environment)
        lock(multi_item_gurobi_environment_locks[thread_id]) do
            if isnothing(multi_item_gurobi_environments[thread_id])
                multi_item_gurobi_environments[thread_id] = Gurobi.Env(
                    Dict{String,Any}("OutputFlag" => 0, "Threads" => 1),
                )
            end
            environment = multi_item_gurobi_environments[thread_id]
        end
    end
    return environment::Gurobi.Env
end


const multi_item_optimizer = optimizer_with_attributes(
    () -> Gurobi.Optimizer(_multi_item_gurobi_environment_for_current_thread()),
)
const multi_item_intersection_geometry_caches = [
    Dict{Any,Tuple{Float64,Vector{Float64}}}()
    for _ in 1:multi_item_julia_thread_count
]


# Each item's newsvendor loss is the maximum of two affine functions. Keeping
# these pieces item-wise avoids enumerating all 2^number_of_items joint pieces.
const multi_item_loss_demand_coefficients =
    [-multi_item_overage_cost, multi_item_underage_cost]
const multi_item_loss_order_coefficients = -multi_item_loss_demand_coefficients
const multi_item_support_matrix = [-1.0, 1.0]
const multi_item_normalized_support_rhs = [0.0, 1.0]
const multi_item_positive_loss_slopes =
    [multi_item_overage_cost, multi_item_underage_cost]
const multi_item_transformed_order_coefficients =
    [multi_item_overage_cost, -multi_item_underage_cost]
const multi_item_transformed_loss_constants = [-multi_item_overage_cost, 0.0]


function _new_multi_item_model()
    problem = Model(multi_item_optimizer)
    set_string_names_on_creation(problem, false)
    return problem
end


function _validate_multi_item_demands(demands)
    isempty(demands) && throw(ArgumentError("at least one demand sample is required"))
    for demand in demands
        length(demand) == multi_item_dimension ||
            throw(DimensionMismatch("each demand sample must contain number_of_items entries"))
        all(isfinite, demand) || throw(ArgumentError("demand samples must be finite"))
    end
    return nothing
end


function _normalized_positive_weights_and_demands(demands, weights)
    _validate_multi_item_demands(demands)
    length(weights) == length(demands) ||
        throw(DimensionMismatch("weights and demands must have the same length"))
    all(isfinite, weights) || throw(ArgumentError("weights must be finite"))
    any(weight -> weight < 0.0, weights) &&
        throw(ArgumentError("weights must be nonnegative"))

    positive_weight_indices = findall(weight -> weight > 0.0, weights)
    isempty(positive_weight_indices) &&
        throw(ArgumentError("at least one weight must be positive"))

    normalized_weights = Float64.(weights[positive_weight_indices])
    normalized_weight_sum = sum(normalized_weights)
    if !isfinite(normalized_weight_sum) || iszero(normalized_weight_sum)
        normalized_weights ./= maximum(normalized_weights)
        normalized_weight_sum = sum(normalized_weights)
        isfinite(normalized_weight_sum) && normalized_weight_sum > 0.0 ||
            throw(ArgumentError("weights could not be normalized safely"))
    end
    normalized_weights ./= normalized_weight_sum
    return demands[positive_weight_indices], normalized_weights
end


function _optimize_multi_item_model!(problem; high_precision = false)
    if !high_precision
        set_attribute(problem, "BarHomogeneous", -1)
        set_attribute(problem, "NumericFocus", 0)
        optimize!(problem)
        is_solved_and_feasible(problem) && return nothing
    end

    set_attribute(problem, "BarHomogeneous", 1)
    set_attribute(problem, "NumericFocus", 3)
    if high_precision
        set_attribute(problem, "FeasibilityTol", 1.0e-9)
        set_attribute(problem, "BarQCPConvTol", 1.0e-10)
    end
    optimize!(problem)
    is_solved_and_feasible(problem) && return nothing

    error(
        "Gurobi did not solve the multi-item newsvendor model: " *
        "termination_status=$(termination_status(problem)), " *
        "primal_status=$(primal_status(problem))",
    )
end


function _minimum_scalar_intersection_epsilon_and_point(
    scalar_demands, relative_radii,
)
    K = length(scalar_demands)
    minimum_epsilon = 0.0
    for k in 1:K
        minimum_epsilon = max(
            minimum_epsilon,
            (scalar_demands[k] - 1.0) / relative_radii[k],
            -scalar_demands[k] / relative_radii[k],
        )
        for j in 1:k-1
            minimum_epsilon = max(
                minimum_epsilon,
                abs(scalar_demands[j] - scalar_demands[k]) /
                (relative_radii[j] + relative_radii[k]),
            )
        end
    end

    feasible_lower_bound = max(
        0.0,
        maximum(
            scalar_demands[k] - relative_radii[k] * minimum_epsilon
            for k in 1:K
        ),
    )
    feasible_upper_bound = min(
        1.0,
        minimum(
            scalar_demands[k] + relative_radii[k] * minimum_epsilon
            for k in 1:K
        ),
    )
    feasible_point = (feasible_lower_bound + feasible_upper_bound) / 2.0
    return minimum_epsilon, [feasible_point]
end


function _compute_minimum_intersection_epsilon_and_point(
    normalized_demands, radius_ratio,
)
    K = length(normalized_demands)
    relative_radii = [
        1.0 + (K - k + 1) * radius_ratio
        for k in 1:K
    ]
    all(isfinite, relative_radii) ||
        throw(ArgumentError("the relative intersection radii must be finite"))

    if multi_item_dimension == 1
        return _minimum_scalar_intersection_epsilon_and_point(
            [normalized_demands[k][1] for k in 1:K], relative_radii,
        )
    end

    geometry_problem = _new_multi_item_model()
    @variables(geometry_problem, begin
        1.0 >= feasible_point[i = 1:multi_item_dimension] >= 0.0
        minimum_normalized_epsilon >= 0.0
    end)
    for k in 1:K
        @constraint(
            geometry_problem,
            [
                relative_radii[k] * minimum_normalized_epsilon;
                [
                    feasible_point[i] - normalized_demands[k][i]
                    for i in 1:multi_item_dimension
                ]
            ] in MathOptInterface.SecondOrderCone(multi_item_dimension + 1),
        )
    end
    @objective(geometry_problem, Min, minimum_normalized_epsilon)
    _optimize_multi_item_model!(geometry_problem; high_precision = true)

    return value(minimum_normalized_epsilon), value.(feasible_point)
end


function _minimum_intersection_epsilon_and_point(normalized_demands, radius_ratio)
    geometry_cache = multi_item_intersection_geometry_caches[Threads.threadid()]
    cache_key = (
        Float64(radius_ratio),
        Tuple(Tuple(demand) for demand in normalized_demands),
    )
    return get!(geometry_cache, cache_key) do
        _compute_minimum_intersection_epsilon_and_point(
            normalized_demands, radius_ratio,
        )
    end
end


function _weighted_newsvendor_quantile(values, weights, probability)
    permutation = sortperm(values)
    cumulative_weight = 0.0
    for index in permutation
        cumulative_weight += weights[index]
        if cumulative_weight >= probability
            return values[index]
        end
    end
    return values[permutation[end]]
end


function SO_multi_item_newsvendor_objective_value_and_order(
    _, demands, weights, doubling_count,
)
    demands, weights = _normalized_positive_weights_and_demands(demands, weights)
    critical_fractile =
        multi_item_underage_cost /
        (multi_item_overage_cost + multi_item_underage_cost)

    # With no coupling constraint between items, a weighted critical-fractile
    # quantile is an exact optimizer for each coordinate.
    order = if iszero(multi_item_underage_cost)
        zeros(multi_item_dimension)
    else
        [
            clamp(
                _weighted_newsvendor_quantile(
                    [demands[t][i] for t in eachindex(demands)],
                    weights,
                    critical_fractile,
                ),
                0.0,
                multi_item_demand_upper_bound,
            ) for i in 1:multi_item_dimension
        ]
    end

    objective = sum(
        weights[t] * sum(
            multi_item_underage_cost * max(demands[t][i] - order[i], 0.0) +
            multi_item_overage_cost * max(order[i] - demands[t][i], 0.0)
            for i in 1:multi_item_dimension
        ) for t in eachindex(demands)
    )

    return objective, order, doubling_count
end


function _conic_W2_DRO_multi_item_newsvendor_objective_value_and_order(
    epsilon, demands, weights, doubling_count,
)
    isfinite(epsilon) && epsilon >= 0.0 ||
        throw(ArgumentError("the ambiguity radius must be finite and nonnegative"))
    if iszero(epsilon)
        return SO_multi_item_newsvendor_objective_value_and_order(
            epsilon, demands, weights, doubling_count,
        )
    end

    demands, weights = _normalized_positive_weights_and_demands(demands, weights)
    normalized_demands = [demand ./ multi_item_demand_upper_bound for demand in demands]
    normalized_epsilon = epsilon / multi_item_demand_upper_bound
    isfinite(normalized_epsilon^2) ||
        throw(ArgumentError("the squared normalized ambiguity radius must be finite"))
    T = length(normalized_demands)
    number_of_loss_pieces = length(multi_item_loss_demand_coefficients)

    # For fixed lambda, both the box support and the squared Euclidean transport
    # cost separate by item. The common lambda preserves the single joint W2
    # budget, while the item-wise epigraphs replace 2^number_of_items joint loss
    # pieces with two scalar rotated cones per sample and item.
    problem = _new_multi_item_model()
    @variables(problem, begin
        1.0 >= normalized_order[i = 1:multi_item_dimension] >= 0.0
        lambda >= 0.0
        gamma[t = 1:T, i = 1:multi_item_dimension]
        z[t = 1:T, i = 1:multi_item_dimension,
          l = 1:number_of_loss_pieces, m = 1:2] >= 0.0
    end)

    for t in 1:T, i in 1:multi_item_dimension, l in 1:number_of_loss_pieces
        demand = normalized_demands[t][i]
        @constraint(
            problem,
            [
                2.0 * lambda;
                gamma[t, i] -
                multi_item_loss_order_coefficients[l] * normalized_order[i] +
                lambda * demand^2 -
                sum(z[t, i, l, m] * multi_item_normalized_support_rhs[m] for m in 1:2);
                multi_item_loss_demand_coefficients[l] + 2.0 * lambda * demand -
                sum(multi_item_support_matrix[m] * z[t, i, l, m] for m in 1:2)
            ] in MathOptInterface.RotatedSecondOrderCone(3),
        )
    end

    @objective(
        problem,
        Min,
        normalized_epsilon^2 * lambda +
        sum(weights[t] * gamma[t, i] for t in 1:T, i in 1:multi_item_dimension),
    )

    _optimize_multi_item_model!(problem)
    objective = multi_item_demand_upper_bound * objective_value(problem)
    order = multi_item_demand_upper_bound .* value.(normalized_order)
    return objective, order, doubling_count
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


function _prepare_weighted_W2_closed_form(demands, weights)
    T = length(demands)
    normalized_demands = Matrix{Float64}(undef, T, multi_item_dimension)
    for t in 1:T, i in 1:multi_item_dimension
        normalized_demands[t, i] = demands[t][i] / multi_item_demand_upper_bound
    end

    all(value -> 0.0 <= value <= 1.0, normalized_demands) || return nothing

    critical_fractile =
        multi_item_underage_cost /
        (multi_item_overage_cost + multi_item_underage_cost)
    quantiles = zeros(multi_item_dimension)
    displacement_terms = _WeightedW2DisplacementTerm[]

    for i in 1:multi_item_dimension
        item_demands = view(normalized_demands, :, i)
        quantile_demand =
            _weighted_newsvendor_quantile(item_demands, weights, critical_fractile)
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
                    displacement_terms,
                    weights[t],
                    demand,
                    multi_item_overage_cost,
                )
            elseif demand > quantile_demand
                _push_weighted_W2_displacement_term!(
                    displacement_terms,
                    weights[t],
                    1.0 - demand,
                    multi_item_underage_cost,
                )
            end
        end

        _push_weighted_W2_displacement_term!(
            displacement_terms,
            tie_overage_mass,
            quantile_demand,
            multi_item_overage_cost,
        )
        _push_weighted_W2_displacement_term!(
            displacement_terms,
            tie_underage_mass,
            1.0 - quantile_demand,
            multi_item_underage_cost,
        )
    end

    sort!(displacement_terms, by = term -> term.threshold)
    return normalized_demands, quantiles, displacement_terms
end


function _weighted_W2_squared_displacement(lambda, displacement_terms)
    return sum(
        (
            lambda < term.threshold ?
            term.saturated_value :
            term.inverse_square_coefficient / lambda^2
            for term in displacement_terms
        );
        init = 0.0,
    )
end


function _weighted_W2_lambda_by_bisection(
    normalized_epsilon_squared, displacement_terms,
)
    lower = 0.0
    upper = isempty(displacement_terms) ? 1.0 : displacement_terms[end].threshold
    while _weighted_W2_squared_displacement(upper, displacement_terms) >
          normalized_epsilon_squared
        upper *= 2.0
    end

    for _ in 1:100
        midpoint = (lower + upper) / 2.0
        if _weighted_W2_squared_displacement(midpoint, displacement_terms) >
           normalized_epsilon_squared
            lower = midpoint
        else
            upper = midpoint
        end
    end
    return (lower + upper) / 2.0
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

    # This is only a roundoff safeguard; convexity guarantees an interval root.
    return _weighted_W2_lambda_by_bisection(
        normalized_epsilon_squared, displacement_terms,
    )
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


function _solve_weighted_W2_closed_form(
    epsilon,
    normalized_demands,
    quantiles,
    displacement_terms,
    weights,
    doubling_count,
)
    normalized_epsilon = epsilon / multi_item_demand_upper_bound
    normalized_epsilon_squared = normalized_epsilon^2
    isfinite(normalized_epsilon_squared) ||
        throw(ArgumentError("the squared normalized ambiguity radius must be finite"))

    lambda = _optimal_weighted_W2_lambda(
        normalized_epsilon_squared, displacement_terms,
    )
    critical_fractile =
        multi_item_underage_cost /
        (multi_item_overage_cost + multi_item_underage_cost)

    normalized_order = if iszero(lambda)
        fill(critical_fractile, multi_item_dimension)
    else
        [
            clamp(
                (
                    _bounded_linear_quadratic_conjugate(
                        multi_item_underage_cost, quantiles[i], lambda,
                    ) -
                    _bounded_linear_quadratic_conjugate(
                        -multi_item_overage_cost, quantiles[i], lambda,
                    )
                ) /
                (multi_item_underage_cost + multi_item_overage_cost),
                0.0,
                1.0,
            ) for i in 1:multi_item_dimension
        ]
    end

    normalized_objective = normalized_epsilon_squared * lambda
    for t in eachindex(weights), i in 1:multi_item_dimension
        demand = normalized_demands[t, i]
        normalized_objective += weights[t] * max(
            -multi_item_underage_cost * normalized_order[i] +
            _bounded_linear_quadratic_conjugate(
                multi_item_underage_cost, demand, lambda,
            ),
            multi_item_overage_cost * normalized_order[i] +
            _bounded_linear_quadratic_conjugate(
                -multi_item_overage_cost, demand, lambda,
            ),
        )
    end

    objective = multi_item_demand_upper_bound * normalized_objective
    order = multi_item_demand_upper_bound .* normalized_order
    return objective, order, doubling_count
end


function W2_DRO_multi_item_newsvendor_objective_value_and_order(
    epsilon, demands, weights, doubling_count,
)
    isfinite(epsilon) && epsilon >= 0.0 ||
        throw(ArgumentError("the ambiguity radius must be finite and nonnegative"))
    if iszero(epsilon)
        return SO_multi_item_newsvendor_objective_value_and_order(
            epsilon, demands, weights, doubling_count,
        )
    end

    demands, weights = _normalized_positive_weights_and_demands(demands, weights)
    closed_form_data = _prepare_weighted_W2_closed_form(demands, weights)
    if isnothing(closed_form_data)
        return _conic_W2_DRO_multi_item_newsvendor_objective_value_and_order(
            epsilon, demands, weights, doubling_count,
        )
    end

    normalized_demands, quantiles, displacement_terms = closed_form_data
    return _solve_weighted_W2_closed_form(
        epsilon,
        normalized_demands,
        quantiles,
        displacement_terms,
        weights,
        doubling_count,
    )
end


function _build_one_sided_intersection_DRO_model(normalized_demands)
    K = length(normalized_demands)
    number_of_loss_pieces = length(multi_item_positive_loss_slopes)
    problem = _new_multi_item_model()
    @variables(problem, begin
        1.0 >= normalized_order[i = 1:multi_item_dimension] >= 0.0
        lambda[k = 1:K] >= 0.0
        radius_penalty >= 0.0
        eta[i = 1:multi_item_dimension]
        z[i = 1:multi_item_dimension, l = 1:number_of_loss_pieces] >= 0.0
    end)

    lambda_sum = sum(lambda[k] for k in 1:K)
    for i in 1:multi_item_dimension, l in 1:number_of_loss_pieces
        weighted_demand = sum(
            lambda[k] * (
                l == 1 ?
                1.0 - normalized_demands[k][i] :
                normalized_demands[k][i]
            ) for k in 1:K
        )
        weighted_squared_demand = sum(
            lambda[k] * (
                l == 1 ?
                1.0 - normalized_demands[k][i] :
                normalized_demands[k][i]
            )^2 for k in 1:K
        )

        @constraint(
            problem,
            [
                2.0 * lambda_sum;
                eta[i] -
                multi_item_transformed_order_coefficients[l] * normalized_order[i] -
                multi_item_transformed_loss_constants[l] +
                weighted_squared_demand - z[i, l];
                multi_item_positive_loss_slopes[l] +
                2.0 * weighted_demand - z[i, l]
            ] in MathOptInterface.RotatedSecondOrderCone(3),
        )
    end

    radius_penalty_constraint = @constraint(
        problem,
        radius_penalty == sum(lambda[k] for k in 1:K),
    )
    @objective(problem, Min, sum(eta))
    return (
        problem,
        normalized_order,
        lambda,
        radius_penalty,
        radius_penalty_constraint,
    )
end


function _build_generic_intersection_DRO_model(normalized_demands)
    K = length(normalized_demands)
    number_of_loss_pieces = length(multi_item_loss_demand_coefficients)
    problem = _new_multi_item_model()
    @variables(problem, begin
        1.0 >= normalized_order[i = 1:multi_item_dimension] >= 0.0
        lambda[k = 1:K] >= 0.0
        radius_penalty >= 0.0
        eta[i = 1:multi_item_dimension]
        z[i = 1:multi_item_dimension, l = 1:number_of_loss_pieces, m = 1:2] >= 0.0
    end)

    lambda_sum = sum(lambda[k] for k in 1:K)
    for i in 1:multi_item_dimension, l in 1:number_of_loss_pieces
        weighted_demand =
            sum(lambda[k] * normalized_demands[k][i] for k in 1:K)
        weighted_squared_demand =
            sum(lambda[k] * normalized_demands[k][i]^2 for k in 1:K)
        @constraint(
            problem,
            [
                2.0 * lambda_sum;
                eta[i] -
                multi_item_loss_order_coefficients[l] * normalized_order[i] +
                weighted_squared_demand -
                sum(z[i, l, m] * multi_item_normalized_support_rhs[m] for m in 1:2);
                multi_item_loss_demand_coefficients[l] +
                2.0 * weighted_demand -
                sum(multi_item_support_matrix[m] * z[i, l, m] for m in 1:2)
            ] in MathOptInterface.RotatedSecondOrderCone(3),
        )
    end

    radius_penalty_constraint = @constraint(
        problem,
        radius_penalty == sum(lambda[k] for k in 1:K),
    )
    @objective(problem, Min, sum(eta))
    return (
        problem,
        normalized_order,
        lambda,
        radius_penalty,
        radius_penalty_constraint,
    )
end


function _build_intersection_DRO_model(normalized_demands)
    if all(
        demand -> all(value -> 0.0 <= value <= 1.0, demand),
        normalized_demands,
    )
        return _build_one_sided_intersection_DRO_model(normalized_demands)
    end
    return _build_generic_intersection_DRO_model(normalized_demands)
end


function _intersection_zero_multiplier_solution(
    normalized_demands, normalized_ball_radii, doubling_count,
)
    K = length(normalized_demands)
    critical_fractile =
        multi_item_underage_cost /
        (multi_item_overage_cost + multi_item_underage_cost)
    is_optimal = all(1:K) do k
        required_squared_radius = sum(
            (1.0 - critical_fractile) *
            (1.0 - normalized_demands[k][i])^2 +
            critical_fractile * normalized_demands[k][i]^2
            for i in 1:multi_item_dimension
        )
        normalized_ball_radii[k]^2 >= required_squared_radius
    end
    is_optimal || return nothing

    normalized_objective =
        multi_item_dimension * multi_item_underage_cost *
        multi_item_overage_cost /
        (multi_item_underage_cost + multi_item_overage_cost)
    return (
        multi_item_demand_upper_bound * normalized_objective,
        fill(
            multi_item_demand_upper_bound * critical_fractile,
            multi_item_dimension,
        ),
        doubling_count,
    )
end


function REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
    epsilon, demands, weights, doubling_count,
)
    isfinite(epsilon) && epsilon >= 0.0 ||
        throw(ArgumentError("the ambiguity radius must be finite and nonnegative"))
    _validate_multi_item_demands(demands)
    K = length(demands)
    length(weights) == K ||
        throw(DimensionMismatch("weights and demands must have the same length"))
    all(isfinite, weights) || throw(ArgumentError("weights must be finite"))
    any(weight -> weight < 0.0, weights) &&
        throw(ArgumentError("intersection radius ratios must be nonnegative"))

    # REMK_intersection_weights carries rho / epsilon in every entry; these are
    # radius parameters, not probability weights, so they must not be normalized.
    normalized_demands = [demand ./ multi_item_demand_upper_bound for demand in demands]
    ball_radii = REMK_intersection_ball_radii(K, epsilon, weights[end])
    all(isfinite, ball_radii) ||
        throw(ArgumentError("the intersection ball radii must be finite"))
    normalized_ball_radii = ball_radii ./ multi_item_demand_upper_bound
    all(radius -> isfinite(radius^2), normalized_ball_radii) ||
        throw(ArgumentError("the squared normalized intersection radii must be finite"))

    if iszero(epsilon)
        first_demand = normalized_demands[1]
        all(demand -> demand == first_demand, normalized_demands) ||
            throw(ArgumentError("zero-radius intersection is empty for distinct demands"))
        all(value -> 0.0 <= value <= 1.0, first_demand) ||
            throw(ArgumentError("zero-radius intersection lies outside the demand support"))
        return SO_multi_item_newsvendor_objective_value_and_order(
            0.0, [demands[1]], [1.0], doubling_count,
        )
    end

    zero_multiplier_solution = _intersection_zero_multiplier_solution(
        normalized_demands, normalized_ball_radii, doubling_count,
    )
    isnothing(zero_multiplier_solution) || return zero_multiplier_solution

    # W2(P, delta_d)^2 = ||E[P] - d||_2^2 + tr(Cov(P)). Thus the
    # distributional balls intersect if and only if these Euclidean balls do.
    # Their radii are homogeneous in epsilon, so cache the minimum feasible
    # epsilon for each demand history and rho / epsilon ratio.
    minimum_normalized_epsilon, feasible_point =
        _minimum_intersection_epsilon_and_point(normalized_demands, weights[end])
    normalized_epsilon = epsilon / multi_item_demand_upper_bound

    if minimum_normalized_epsilon >= normalized_epsilon
        # Preserve the scalar implementation's convention: at (or beyond) the
        # first radius scaling for which the balls touch, use that point mass.
        # This also supplies a defined fallback when the requested intersection
        # itself is empty.
        touching_demand = multi_item_demand_upper_bound .* feasible_point
        return SO_multi_item_newsvendor_objective_value_and_order(
            0.0, [touching_demand], [1.0], doubling_count,
        )
    end

    problem, normalized_order, lambda, radius_penalty, radius_penalty_constraint =
        _build_intersection_DRO_model(normalized_demands)
    set_normalized_coefficient(
        fill(radius_penalty_constraint, K),
        collect(lambda),
        -normalized_ball_radii .^ 2,
    )
    set_objective_coefficient(problem, radius_penalty, 1.0)

    _optimize_multi_item_model!(problem)
    objective = multi_item_demand_upper_bound * objective_value(problem)
    order = multi_item_demand_upper_bound .* value.(normalized_order)
    return objective, order, doubling_count
end


function _multi_item_newsvendor_grid(
    newsvendor_objective_value_and_order,
    ambiguity_radii,
    demands,
    weight_vectors,
    doubling_count,
)
    result_type = Tuple{Float64,Vector{Float64},typeof(doubling_count)}
    results = Matrix{result_type}(
        undef, length(ambiguity_radii), length(weight_vectors),
    )
    for weight_index in eachindex(weight_vectors), radius_index in eachindex(ambiguity_radii)
        results[radius_index, weight_index] = newsvendor_objective_value_and_order(
            ambiguity_radii[radius_index],
            demands,
            weight_vectors[weight_index],
            doubling_count,
        )
    end
    return results
end


function _multi_item_newsvendor_grid(
    ::typeof(W2_DRO_multi_item_newsvendor_objective_value_and_order),
    ambiguity_radii,
    demands,
    weight_vectors,
    doubling_count,
)
    result_type = Tuple{Float64,Vector{Float64},typeof(doubling_count)}
    results = Matrix{result_type}(
        undef, length(ambiguity_radii), length(weight_vectors),
    )
    for weight_index in eachindex(weight_vectors)
        active_demands, normalized_weights =
            _normalized_positive_weights_and_demands(
                demands, weight_vectors[weight_index],
            )
        closed_form_data = _prepare_weighted_W2_closed_form(
            active_demands, normalized_weights,
        )

        for radius_index in eachindex(ambiguity_radii)
            epsilon = ambiguity_radii[radius_index]
            isfinite(epsilon) && epsilon >= 0.0 ||
                throw(ArgumentError("the ambiguity radius must be finite and nonnegative"))
            if iszero(epsilon)
                results[radius_index, weight_index] =
                    SO_multi_item_newsvendor_objective_value_and_order(
                        epsilon,
                        active_demands,
                        normalized_weights,
                        doubling_count,
                    )
            elseif isnothing(closed_form_data)
                results[radius_index, weight_index] =
                    _conic_W2_DRO_multi_item_newsvendor_objective_value_and_order(
                        epsilon,
                        active_demands,
                        normalized_weights,
                        doubling_count,
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
                        doubling_count,
                    )
            end
        end
    end
    return results
end


function _multi_item_newsvendor_grid(
    ::typeof(REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order),
    ambiguity_radii,
    demands,
    weight_vectors,
    doubling_count,
)
    _validate_multi_item_demands(demands)
    K = length(demands)
    normalized_demands = [demand ./ multi_item_demand_upper_bound for demand in demands]
    result_type = Tuple{Float64,Vector{Float64},typeof(doubling_count)}
    results = Matrix{result_type}(
        undef, length(ambiguity_radii), length(weight_vectors),
    )

    problem = nothing
    normalized_order = nothing
    lambda = nothing
    radius_penalty = nothing
    radius_penalty_constraint = nothing

    for weight_index in eachindex(weight_vectors)
        weights = weight_vectors[weight_index]
        length(weights) == K ||
            throw(DimensionMismatch("weights and demands must have the same length"))
        all(isfinite, weights) || throw(ArgumentError("weights must be finite"))
        any(weight -> weight < 0.0, weights) &&
            throw(ArgumentError("intersection radius ratios must be nonnegative"))

        radius_ratio = weights[end]
        relative_radii = [
            1.0 + (K - k + 1) * radius_ratio
            for k in 1:K
        ]
        all(isfinite, relative_radii) ||
            throw(ArgumentError("the relative intersection radii must be finite"))
        relative_radius_scale = maximum(relative_radii)
        scaled_relative_radii = relative_radii ./ relative_radius_scale

        geometry = nothing
        radius_penalty_is_current = false
        for radius_index in eachindex(ambiguity_radii)
            epsilon = ambiguity_radii[radius_index]
            isfinite(epsilon) && epsilon >= 0.0 ||
                throw(ArgumentError("the ambiguity radius must be finite and nonnegative"))

            if iszero(epsilon)
                results[radius_index, weight_index] =
                    REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
                        epsilon, demands, weights, doubling_count,
                    )
                continue
            end

            normalized_epsilon = epsilon / multi_item_demand_upper_bound
            normalized_ball_radii = normalized_epsilon .* relative_radii
            all(radius -> isfinite(radius^2), normalized_ball_radii) ||
                throw(ArgumentError("the squared normalized intersection radii must be finite"))

            zero_multiplier_solution = _intersection_zero_multiplier_solution(
                normalized_demands, normalized_ball_radii, doubling_count,
            )
            if !isnothing(zero_multiplier_solution)
                results[radius_index, weight_index] = zero_multiplier_solution
                continue
            end

            if isnothing(geometry)
                geometry = _compute_minimum_intersection_epsilon_and_point(
                    normalized_demands, radius_ratio,
                )
            end
            minimum_normalized_epsilon, feasible_point = geometry
            if minimum_normalized_epsilon >= normalized_epsilon
                touching_demand = multi_item_demand_upper_bound .* feasible_point
                results[radius_index, weight_index] =
                    SO_multi_item_newsvendor_objective_value_and_order(
                        0.0, [touching_demand], [1.0], doubling_count,
                    )
                continue
            end

            if isnothing(problem)
                problem,
                normalized_order,
                lambda,
                radius_penalty,
                radius_penalty_constraint =
                    _build_intersection_DRO_model(normalized_demands)
            end
            if !radius_penalty_is_current
                set_normalized_coefficient(
                    fill(radius_penalty_constraint, K),
                    collect(lambda),
                    -scaled_relative_radii .^ 2,
                )
                radius_penalty_is_current = true
            end
            set_objective_coefficient(
                problem,
                radius_penalty,
                (normalized_epsilon * relative_radius_scale)^2,
            )
            _optimize_multi_item_model!(problem)
            results[radius_index, weight_index] = (
                multi_item_demand_upper_bound * objective_value(problem),
                multi_item_demand_upper_bound .* value.(normalized_order),
                doubling_count,
            )
        end
    end

    return results
end
