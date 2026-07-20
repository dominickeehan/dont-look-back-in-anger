# Intersection-specific geometry, solver shortcuts, and grid execution.
# This file is included at the end of multi-item-newsvendor-optimizations.jl
# and uses the shared model, SAA, and weighted-W2 functions defined there.

const additive_intersection_geometry_caches = [Dict{Any,Tuple{Float64,Vector{Float64}}}() for _ in 1:julia_thread_count]
const minimax_center_caches = [Dict{Any,Tuple{Float64,Vector{Float64}}}() for _ in 1:julia_thread_count]

const multi_item_intersection_dual_relative_gap_tolerance = 1.0e-6
const multi_item_intersection_dual_absolute_gap_tolerance = 1.0e-8
const multi_item_intersection_dual_max_iterations = 2000
const multi_item_enable_intersection_dual_solver = Ref(true)
const multi_item_pair_certificate_relative_gap_tolerance = 1.0e-12
const multi_item_pair_certificate_absolute_gap_tolerance = 1.0e-14
const multi_item_geometry_comparison_relative_tolerance = 1.0e-9
const multi_item_geometry_comparison_absolute_tolerance = 1.0e-10


function _intersection_geometry_threshold_relation(minimum_value, requested_value)
    tolerance = max(
        multi_item_geometry_comparison_absolute_tolerance,
        multi_item_geometry_comparison_relative_tolerance * max(
            abs(minimum_value), abs(requested_value),
        ),
    )
    minimum_value > requested_value + tolerance && return 1
    minimum_value < requested_value - tolerance && return -1
    return 0
end


Base.@kwdef mutable struct _MultiItemSolverStatistics
    touching_solutions::Int = 0
    additive_radius_repairs::Int = 0
    zero_multiplier_solutions::Int = 0
    single_ball_solutions::Int = 0
    dual_solver_solutions::Int = 0
    dual_solver_failures::Int = 0
    conic_solutions::Int = 0
    numeric_retry_solves::Int = 0
    additive_geometry_solves::Int = 0
    additive_candidate_certificate_solutions::Int = 0
    additive_geometry_socp_solves::Int = 0
    active_ball_count_sum::Int = 0
    total_ball_count_sum::Int = 0
    pruned_solve_observations::Int = 0
end


const multi_item_solver_statistics = [_MultiItemSolverStatistics() for _ in 1:julia_thread_count]

_multi_item_statistics() = multi_item_solver_statistics[Threads.threadid()]


function multi_item_reset_solver_statistics!()
    for thread_index in eachindex(multi_item_solver_statistics)
        multi_item_solver_statistics[thread_index] = _MultiItemSolverStatistics()
    end
    return nothing
end


function multi_item_solver_statistics_summary()
    aggregate = _MultiItemSolverStatistics()
    for statistics in multi_item_solver_statistics
        for field in fieldnames(_MultiItemSolverStatistics)
            setfield!(
                aggregate,
                field,
                getfield(aggregate, field) + getfield(statistics, field),
            )
        end
    end
    mean_active_balls = aggregate.pruned_solve_observations > 0 ?
        aggregate.active_ball_count_sum / aggregate.pruned_solve_observations :
        NaN
    mean_total_balls = aggregate.pruned_solve_observations > 0 ?
        aggregate.total_ball_count_sum / aggregate.pruned_solve_observations :
        NaN
    return (
        touching_solutions = aggregate.touching_solutions,
        additive_radius_repairs = aggregate.additive_radius_repairs,
        zero_multiplier_solutions = aggregate.zero_multiplier_solutions,
        single_ball_solutions = aggregate.single_ball_solutions,
        dual_solver_solutions = aggregate.dual_solver_solutions,
        dual_solver_failures = aggregate.dual_solver_failures,
        conic_solutions = aggregate.conic_solutions,
        numeric_retry_solves = aggregate.numeric_retry_solves,
        additive_geometry_solves = aggregate.additive_geometry_solves,
        additive_candidate_certificate_solutions =
            aggregate.additive_candidate_certificate_solutions,
        additive_geometry_socp_solves = aggregate.additive_geometry_socp_solves,
        mean_active_balls = mean_active_balls,
        mean_total_balls = mean_total_balls,
    )
end


function _geometry_constraining_ball_indices(
    normalized_demands, relative_radii, epsilon_lower_bound,
)
    permutation = sortperm(relative_radii)
    kept = Int[]
    for j in permutation
        contained = false
        for k in kept
            radius_gap = (relative_radii[j] - relative_radii[k]) * epsilon_lower_bound
            if radius_gap >= 0.0 &&
               _squared_euclidean_distance(
                   normalized_demands[j], normalized_demands[k],
               ) <= radius_gap * radius_gap
                contained = true
                break
            end
        end
        contained || push!(kept, j)
    end
    return sort!(kept)
end


function _required_additive_intersection_rho_at_point(
    point, normalized_demands, normalized_epsilon,
)
    K = length(normalized_demands)
    required_rho = 0.0
    for k in 1:K
        distance = sqrt(_squared_euclidean_distance(point, normalized_demands[k]))
        required_rho = max(
            required_rho,
            (distance - normalized_epsilon) / (K - k + 1),
        )
    end
    return required_rho
end


function _additive_intersection_rho_lower_bound_and_candidate(
    normalized_demands, normalized_epsilon,
)
    K = length(normalized_demands)
    lower_bound = 0.0
    first_index = 0
    second_index = 0

    for k in 1:K
        projected = clamp.(normalized_demands[k], 0.0, 1.0)
        support_distance = sqrt(
            _squared_euclidean_distance(normalized_demands[k], projected),
        )
        bound = (support_distance - normalized_epsilon) / (K - k + 1)
        if bound > lower_bound
            lower_bound = bound
            first_index = k
            second_index = 0
        end
    end

    for j in 1:K, k in j+1:K
        center_distance = sqrt(
            _squared_euclidean_distance(normalized_demands[j], normalized_demands[k]),
        )
        coefficient_sum = (K - j + 1) + (K - k + 1)
        bound =
            (center_distance - 2.0 * normalized_epsilon) / coefficient_sum
        if bound > lower_bound
            lower_bound = bound
            first_index = j
            second_index = k
        end
    end
    return lower_bound, first_index, second_index
end


function _additive_intersection_candidate_certificate(
    normalized_demands,
    normalized_epsilon,
    rho_lower_bound,
    first_index,
    second_index,
)
    first_index > 0 || return nothing
    candidate = if second_index == 0
        clamp.(normalized_demands[first_index], 0.0, 1.0)
    else
        center_distance = sqrt(
            _squared_euclidean_distance(
                normalized_demands[first_index],
                normalized_demands[second_index],
            ),
        )
        center_distance > 0.0 || return nothing
        first_radius =
            normalized_epsilon +
            (length(normalized_demands) - first_index + 1) * rho_lower_bound
        fraction = first_radius / center_distance
        normalized_demands[first_index] .+ fraction .* (
            normalized_demands[second_index] .- normalized_demands[first_index]
        )
    end
    all(value -> 0.0 <= value <= 1.0, candidate) || return nothing

    rho_upper_bound = _required_additive_intersection_rho_at_point(
        candidate, normalized_demands, normalized_epsilon,
    )
    certificate_tolerance = max(
        multi_item_pair_certificate_absolute_gap_tolerance,
        multi_item_pair_certificate_relative_gap_tolerance * max(
            abs(rho_lower_bound), abs(rho_upper_bound),
        ),
    )
    rho_upper_bound <= rho_lower_bound + certificate_tolerance || return nothing
    return max(rho_lower_bound, rho_upper_bound), candidate
end


function _additive_intersection_candidate_upper_bound_and_point(
    normalized_demands, normalized_epsilon,
)
    K = length(normalized_demands)
    mean_point = zeros(number_of_items)
    for demand in normalized_demands
        mean_point .+= demand
    end
    mean_point ./= K
    clamp!(mean_point, 0.0, 1.0)

    best_rho = Inf
    best_point = mean_point
    candidate = similar(mean_point)
    for candidate_index in 0:K
        if candidate_index == 0
            candidate .= mean_point
        else
            candidate .= normalized_demands[candidate_index]
            clamp!(candidate, 0.0, 1.0)
        end
        required_rho = _required_additive_intersection_rho_at_point(
            candidate, normalized_demands, normalized_epsilon,
        )
        if required_rho < best_rho
            best_rho = required_rho
            best_point = copy(candidate)
        end
    end
    return best_rho, best_point
end


function _compute_minimum_additive_intersection_rho_and_point(
    normalized_demands,
    normalized_epsilon,
)
    _multi_item_statistics().additive_geometry_solves += 1

    K = length(normalized_demands)
    coefficients = Float64[K - k + 1 for k in 1:K]
    rho_lower_bound, first_index, second_index =
        _additive_intersection_rho_lower_bound_and_candidate(
            normalized_demands, normalized_epsilon,
        )
    certificate = _additive_intersection_candidate_certificate(
        normalized_demands,
        normalized_epsilon,
        rho_lower_bound,
        first_index,
        second_index,
    )
    if !isnothing(certificate)
        _multi_item_statistics().additive_candidate_certificate_solutions += 1
        return certificate
    end

    candidate_upper_bound, candidate_point =
        _additive_intersection_candidate_upper_bound_and_point(
            normalized_demands, normalized_epsilon,
        )
    candidate_tolerance = max(
        multi_item_pair_certificate_absolute_gap_tolerance,
        multi_item_pair_certificate_relative_gap_tolerance * max(
            abs(rho_lower_bound), abs(candidate_upper_bound),
        ),
    )
    if candidate_upper_bound <= rho_lower_bound + candidate_tolerance
        return max(rho_lower_bound, candidate_upper_bound), candidate_point
    end

    constraining_indices = _geometry_constraining_ball_indices(
        normalized_demands, coefficients, rho_lower_bound,
    )
    _multi_item_statistics().additive_geometry_socp_solves += 1
    geometry_problem = _new_multi_item_model()
    @variables(geometry_problem, begin
        1.0 >= feasible_point[i = 1:number_of_items] >= 0.0
        minimum_normalized_rho >= 0.0
    end)
    for k in constraining_indices
        @constraint(
            geometry_problem,
            [
                normalized_epsilon + coefficients[k] * minimum_normalized_rho;
                [
                    feasible_point[i] - normalized_demands[k][i]
                    for i in 1:number_of_items
                ]
            ] in MathOptInterface.SecondOrderCone(number_of_items + 1),
        )
    end
    @objective(geometry_problem, Min, minimum_normalized_rho)
    _optimize_multi_item_model!(geometry_problem; high_precision = true)

    point = clamp.(value.(feasible_point), 0.0, 1.0)
    required_rho = _required_additive_intersection_rho_at_point(
        point, normalized_demands, normalized_epsilon,
    )
    return max(rho_lower_bound, value(minimum_normalized_rho), required_rho), point
end


function _minimum_additive_intersection_rho_and_point(
    normalized_demands, normalized_epsilon,
)
    geometry_cache = additive_intersection_geometry_caches[Threads.threadid()]
    cache_key = (
        Float64(normalized_epsilon),
        Tuple(Tuple(demand) for demand in normalized_demands),
    )
    return get!(geometry_cache, cache_key) do
        _compute_minimum_additive_intersection_rho_and_point(
            normalized_demands, normalized_epsilon,
        )
    end
end


# When the requested and minimum additive rho are both zero, every ball
# carries the common radius epsilon and the additive geometry cannot separate
# tangency from a full interior. The constrained 1-center of the demands
# decides: the intersection has interior exactly when the minimax distance is
# below epsilon. The geometry is epsilon-independent, so it is cached per
# demand history.
function _compute_minimax_distance_and_center(normalized_demands)
    K = length(normalized_demands)

    lower_bound = 0.0
    best_first = 0
    best_second = 0
    for k in 1:K
        projected = clamp.(normalized_demands[k], 0.0, 1.0)
        support_distance = sqrt(
            _squared_euclidean_distance(normalized_demands[k], projected),
        )
        lower_bound = max(lower_bound, support_distance)
    end
    for j in 1:K, k in j+1:K
        pair_bound = 0.5 * sqrt(
            _squared_euclidean_distance(normalized_demands[j], normalized_demands[k]),
        )
        if pair_bound > lower_bound
            lower_bound = pair_bound
            best_first = j
            best_second = k
        end
    end

    mean_point = zeros(number_of_items)
    for demand in normalized_demands
        mean_point .+= demand
    end
    mean_point ./= K
    clamp!(mean_point, 0.0, 1.0)

    best_radius = Inf
    best_point = mean_point
    candidate = similar(mean_point)
    for candidate_index in 0:K+1
        if candidate_index == 0
            candidate .= mean_point
        elseif candidate_index <= K
            candidate .= normalized_demands[candidate_index]
            clamp!(candidate, 0.0, 1.0)
        elseif best_first > 0
            candidate .= 0.5 .* (
                normalized_demands[best_first] .+ normalized_demands[best_second]
            )
            clamp!(candidate, 0.0, 1.0)
        else
            continue
        end
        candidate_radius = 0.0
        for k in 1:K
            candidate_radius = max(
                candidate_radius,
                sqrt(_squared_euclidean_distance(candidate, normalized_demands[k])),
            )
        end
        if candidate_radius < best_radius
            best_radius = candidate_radius
            best_point = copy(candidate)
        end
    end

    certificate_tolerance = max(
        multi_item_pair_certificate_absolute_gap_tolerance,
        multi_item_pair_certificate_relative_gap_tolerance * max(
            abs(lower_bound), abs(best_radius),
        ),
    )
    if best_radius <= lower_bound + certificate_tolerance
        return max(lower_bound, best_radius), best_point
    end

    problem = _new_multi_item_model()
    @variables(problem, begin
        1.0 >= center[i = 1:number_of_items] >= 0.0
        minimax_distance >= 0.0
    end)
    for k in 1:K
        @constraint(
            problem,
            [
                minimax_distance;
                [center[i] - normalized_demands[k][i] for i in 1:number_of_items]
            ] in MathOptInterface.SecondOrderCone(number_of_items + 1),
        )
    end
    @objective(problem, Min, minimax_distance)
    _optimize_multi_item_model!(problem; high_precision = true)
    return max(lower_bound, value(minimax_distance)),
        clamp.(value.(center), 0.0, 1.0)
end


function _minimax_distance_and_center(normalized_demands)
    cache = minimax_center_caches[Threads.threadid()]
    cache_key = Tuple(Tuple(demand) for demand in normalized_demands)
    return get!(cache, cache_key) do
        _compute_minimax_distance_and_center(normalized_demands)
    end
end


function _squared_euclidean_distance(first_point, second_point)
    total = 0.0
    for i in eachindex(first_point)
        difference = first_point[i] - second_point[i]
        total += difference * difference
    end
    return total
end


function _intersection_ball_is_vacuous(normalized_demand, normalized_radius)
    farthest_squared_distance = 0.0
    for value in normalized_demand
        farthest = max(abs(value), abs(value - 1.0))
        farthest_squared_distance += farthest * farthest
    end
    return normalized_radius * normalized_radius >= farthest_squared_distance
end


function _active_intersection_ball_indices(normalized_demands, normalized_ball_radii)
    permutation = sortperm(normalized_ball_radii)
    active = Int[]
    for j in permutation
        _intersection_ball_is_vacuous(
            normalized_demands[j], normalized_ball_radii[j],
        ) && continue
        contained = false
        for k in active
            radius_gap = normalized_ball_radii[j] - normalized_ball_radii[k]
            if radius_gap >= 0.0 &&
               _squared_euclidean_distance(
                   normalized_demands[j], normalized_demands[k],
               ) <= radius_gap * radius_gap
                contained = true
                break
            end
        end
        contained || push!(active, j)
    end
    return sort!(active)
end


function _zero_multiplier_epsilon_threshold(
    normalized_demands,
    relative_radii,
    instance_underage_costs,
    instance_overage_costs,
)
    critical_fractiles =
        instance_underage_costs ./
        (instance_underage_costs .+ instance_overage_costs)
    threshold = 0.0
    for k in eachindex(normalized_demands)
        required_squared_radius = sum(
            (1.0 - critical_fractiles[i]) *
            (1.0 - normalized_demands[k][i])^2 +
            critical_fractiles[i] * normalized_demands[k][i]^2
            for i in 1:number_of_items
        )
        threshold = max(threshold, sqrt(required_squared_radius) / relative_radii[k])
    end
    return threshold
end


function _build_intersection_DRO_model(
    normalized_demands, instance_underage_costs, instance_overage_costs,
)
    K = length(normalized_demands)
    number_of_loss_pieces = 2
    problem = _new_multi_item_model()
    @variables(problem, begin
        1.0 >= normalized_order[i = 1:number_of_items] >= 0.0
        lambda[k = 1:K] >= 0.0
        radius_penalty >= 0.0
        eta[i = 1:number_of_items]
        z[i = 1:number_of_items, l = 1:number_of_loss_pieces] >= 0.0
    end)

    lambda_sum = sum(lambda[k] for k in 1:K)
    for i in 1:number_of_items, l in 1:number_of_loss_pieces
        positive_loss_slope =
            l == 1 ?
            instance_overage_costs[i] :
            instance_underage_costs[i]
        transformed_order_coefficient =
            l == 1 ?
            instance_overage_costs[i] :
            -instance_underage_costs[i]
        transformed_loss_constant =
            l == 1 ? -instance_overage_costs[i] : 0.0
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
                transformed_order_coefficient * normalized_order[i] -
                transformed_loss_constant +
                weighted_squared_demand - z[i, l];
                positive_loss_slope +
                2.0 * weighted_demand - z[i, l]
            ] in MathOptInterface.RotatedSecondOrderCone(3),
        )
    end

    radius_penalty_constraint =
        @constraint(problem, radius_penalty == sum(lambda[k] for k in 1:K))
    @objective(problem, Min, sum(eta))
    return problem, normalized_order, lambda, radius_penalty, radius_penalty_constraint
end


function _zero_multiplier_solution(
    instance_underage_costs, instance_overage_costs,
)
    normalized_order =
        instance_underage_costs ./
        (instance_underage_costs .+ instance_overage_costs)
    normalized_objective = sum(
        max(
            instance_underage_costs[i] * (1.0 - normalized_order[i]),
            instance_overage_costs[i] * normalized_order[i],
        )
        for i in 1:number_of_items
    )
    return (
        number_of_consumers * normalized_objective,
        number_of_consumers .* normalized_order,
    )
end


function _intersection_zero_multiplier_solution(
    normalized_demands,
    normalized_ball_radii,
    instance_underage_costs,
    instance_overage_costs,
)
    K = length(normalized_demands)
    critical_fractiles =
        instance_underage_costs ./
        (instance_underage_costs .+ instance_overage_costs)
    is_optimal = all(1:K) do k
        required_squared_radius = sum(
            (1.0 - critical_fractiles[i]) *
            (1.0 - normalized_demands[k][i])^2 +
            critical_fractiles[i] * normalized_demands[k][i]^2
            for i in 1:number_of_items
        )
        normalized_ball_radii[k]^2 >= required_squared_radius
    end
    is_optimal || return nothing
    return _zero_multiplier_solution(
        instance_underage_costs,
        instance_overage_costs,
    )
end


function _single_ball_intersection_solution(
    normalized_demand,
    normalized_radius,
    instance_underage_costs,
    instance_overage_costs,
)
    demand = number_of_consumers .* normalized_demand
    epsilon = number_of_consumers * normalized_radius
    return W2_DRO_multi_item_newsvendor_objective_value_and_order(
        epsilon,
        [demand],
        [1.0],
        instance_underage_costs,
        instance_overage_costs,
    )
end


function _intersection_dual_objective_and_gradient!(
    gradient,
    atom_upper,
    atom_lower,
    weight_upper,
    weight_lower,
    order,
    lambda,
    demand_matrix,
    squared_radii,
    instance_underage_costs,
    instance_overage_costs,
)
    K, n = size(demand_matrix)
    total_multiplier = 0.0
    objective = 0.0
    for k in 1:K
        total_multiplier += lambda[k]
        objective += lambda[k] * squared_radii[k]
        gradient[k] = squared_radii[k]
    end

    for i in 1:n
        underage = instance_underage_costs[i]
        overage = instance_overage_costs[i]
        if total_multiplier > 0.0
            weighted_center = 0.0
            weighted_squared_center = 0.0
            for k in 1:K
                weighted_center += lambda[k] * demand_matrix[k, i]
                weighted_squared_center += lambda[k] * demand_matrix[k, i]^2
            end
            center = clamp(weighted_center / total_multiplier, 0.0, 1.0)
            upper_displacement = min(1.0 - center, underage / (2.0 * total_multiplier))
            underage_conjugate =
                underage * center + underage * upper_displacement -
                total_multiplier * upper_displacement^2
            atom_upper[i] = center + upper_displacement
            lower_displacement = min(center, overage / (2.0 * total_multiplier))
            overage_conjugate =
                -overage * (center - lower_displacement) -
                total_multiplier * lower_displacement^2
            atom_lower[i] = center - lower_displacement
            recentering_value =
                total_multiplier * center^2 - weighted_squared_center
        else
            underage_conjugate = underage
            atom_upper[i] = 1.0
            overage_conjugate = 0.0
            atom_lower[i] = 0.0
            recentering_value = 0.0
        end

        unclamped_order = (underage_conjugate - overage_conjugate) / (underage + overage)
        order[i] = clamp(unclamped_order, 0.0, 1.0)
        objective += max(
            -underage * order[i] + underage_conjugate,
            overage * order[i] + overage_conjugate,
        ) + recentering_value

        if unclamped_order <= 0.0
            weight_upper[i] = 0.0
            weight_lower[i] = 1.0
        elseif unclamped_order >= 1.0
            weight_upper[i] = 1.0
            weight_lower[i] = 0.0
        else
            weight_upper[i] = overage / (underage + overage)
            weight_lower[i] = underage / (underage + overage)
        end

        for k in 1:K
            upper_difference = atom_upper[i] - demand_matrix[k, i]
            lower_difference = atom_lower[i] - demand_matrix[k, i]
            gradient[k] -=
                weight_upper[i] * upper_difference^2 +
                weight_lower[i] * lower_difference^2
        end
    end
    return objective
end


function _certified_intersection_dual_solution(
    objective,
    lambda,
    demand_matrix,
    squared_radii,
    atom_upper,
    atom_lower,
    weight_upper,
    weight_lower,
    order,
    interior_point,
    instance_underage_costs,
    instance_overage_costs,
)
    K, n = size(demand_matrix)

    contraction = 0.0
    for k in 1:K
        moment = 0.0
        interior_squared_distance = 0.0
        for i in 1:n
            upper_difference = atom_upper[i] - demand_matrix[k, i]
            lower_difference = atom_lower[i] - demand_matrix[k, i]
            moment +=
                weight_upper[i] * upper_difference^2 +
                weight_lower[i] * lower_difference^2
            interior_difference = interior_point[i] - demand_matrix[k, i]
            interior_squared_distance += interior_difference^2
        end
        violation = moment - squared_radii[k]
        if violation > 0.0
            margin = squared_radii[k] - interior_squared_distance
            margin > 0.0 || return nothing
            contraction = max(contraction, violation / margin)
        end
    end
    contraction < 1.0 || return nothing

    lower_bound = 0.0
    for i in 1:n
        lower_bound +=
            (atom_upper[i] - atom_lower[i]) *
            min(
                weight_upper[i] * instance_underage_costs[i],
                weight_lower[i] * instance_overage_costs[i],
            )
    end
    lower_bound *= 1.0 - contraction

    gap = objective - lower_bound
    tolerance =
        multi_item_intersection_dual_absolute_gap_tolerance +
        multi_item_intersection_dual_relative_gap_tolerance * abs(objective)
    gap <= tolerance || return nothing
    return objective, copy(order)
end


function _initial_intersection_dual_lambda(
    active_normalized_demands,
    squared_radii,
    instance_underage_costs,
    instance_overage_costs,
)
    lambda = zeros(length(squared_radii))
    smallest_index = argmin(squared_radii)
    closed_form_data = _prepare_weighted_W2_closed_form(
        [number_of_consumers .* active_normalized_demands[smallest_index]],
        [1.0],
        instance_underage_costs,
        instance_overage_costs,
    )
    _, _, displacement_terms = closed_form_data
    lambda[smallest_index] =
        _optimal_weighted_W2_lambda(squared_radii[smallest_index], displacement_terms)
    return lambda
end


function _solve_intersection_dual(
    active_normalized_demands,
    active_normalized_ball_radii,
    interior_point,
    initial_lambda,
    instance_underage_costs,
    instance_overage_costs,
)
    K = length(active_normalized_demands)
    n = number_of_items
    demand_matrix = Matrix{Float64}(undef, K, n)
    for k in 1:K, i in 1:n
        demand_matrix[k, i] = active_normalized_demands[k][i]
    end
    squared_radii = active_normalized_ball_radii .^ 2

    lambda = max.(initial_lambda, 0.0)
    trial_lambda = similar(lambda)
    gradient = similar(lambda)
    trial_gradient = similar(lambda)
    direction = similar(lambda)
    atom_upper = zeros(n)
    atom_lower = zeros(n)
    weight_upper = zeros(n)
    weight_lower = zeros(n)
    order = zeros(n)

    objective = _intersection_dual_objective_and_gradient!(
        gradient, atom_upper, atom_lower, weight_upper, weight_lower, order,
        lambda,
        demand_matrix,
        squared_radii,
        instance_underage_costs,
        instance_overage_costs,
    )

    history_length = 10
    objective_history = fill(-Inf, history_length)
    objective_history[1] = objective
    history_index = 1
    step_scale = 1.0 / max(1.0e-12, sqrt(sum(abs2, gradient)))

    for _ in 1:multi_item_intersection_dual_max_iterations
        certified = _certified_intersection_dual_solution(
            objective, lambda, demand_matrix, squared_radii,
            atom_upper, atom_lower, weight_upper, weight_lower, order,
            interior_point,
            instance_underage_costs,
            instance_overage_costs,
        )
        if !isnothing(certified)
            certified_objective, certified_order = certified
            return certified_objective, certified_order, copy(lambda)
        end

        directional_derivative = 0.0
        for k in 1:K
            direction[k] = max(0.0, lambda[k] - step_scale * gradient[k]) - lambda[k]
            directional_derivative += gradient[k] * direction[k]
        end
        directional_derivative < -1.0e-18 * (1.0 + abs(objective)) || return nothing

        step = 1.0
        reference_objective = maximum(objective_history)
        accepted = false
        trial_objective = Inf
        for _ in 1:40
            for k in 1:K
                trial_lambda[k] = lambda[k] + step * direction[k]
            end
            trial_objective = _intersection_dual_objective_and_gradient!(
                trial_gradient, atom_upper, atom_lower,
                weight_upper, weight_lower, order,
                trial_lambda,
                demand_matrix,
                squared_radii,
                instance_underage_costs,
                instance_overage_costs,
            )
            if trial_objective <=
               reference_objective + 1.0e-4 * step * directional_derivative
                accepted = true
                break
            end
            step /= 2.0
        end
        accepted || return nothing

        step_inner_product = 0.0
        squared_step_norm = 0.0
        for k in 1:K
            step_difference = trial_lambda[k] - lambda[k]
            gradient_difference = trial_gradient[k] - gradient[k]
            step_inner_product += step_difference * gradient_difference
            squared_step_norm += step_difference^2
            lambda[k] = trial_lambda[k]
            gradient[k] = trial_gradient[k]
        end
        objective = trial_objective
        history_index = history_index % history_length + 1
        objective_history[history_index] = objective
        step_scale = step_inner_product > 1.0e-300 ?
            clamp(squared_step_norm / step_inner_product, 1.0e-12, 1.0e12) :
            min(step_scale * 10.0, 1.0e12)
    end

    certified = _certified_intersection_dual_solution(
        objective, lambda, demand_matrix, squared_radii,
        atom_upper, atom_lower, weight_upper, weight_lower, order,
        interior_point,
        instance_underage_costs,
        instance_overage_costs,
    )
    isnothing(certified) && return nothing
    certified_objective, certified_order = certified
    return certified_objective, certified_order, copy(lambda)
end


function REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
    epsilon,
    demands,
    weights,
    instance_underage_costs,
    instance_overage_costs,
)
    K = length(demands)

    normalized_demands = [demand ./ number_of_consumers for demand in demands]
    ball_radii = REMK_intersection_ball_radii(K, epsilon, weights[end])
    normalized_ball_radii = ball_radii ./ number_of_consumers

    zero_multiplier_solution = _intersection_zero_multiplier_solution(
        normalized_demands,
        normalized_ball_radii,
        instance_underage_costs,
        instance_overage_costs,
    )
    if !isnothing(zero_multiplier_solution)
        _multi_item_statistics().zero_multiplier_solutions += 1
        return zero_multiplier_solution
    end

    # The intersection at (epsilon, rho) is nonempty exactly when the minimum
    # additive rho at this epsilon does not exceed the requested rho. At or
    # below that threshold the ambiguity set collapses to the point mass at
    # the additive first-contact point.
    normalized_epsilon = epsilon / number_of_consumers
    minimum_normalized_rho, feasible_point =
        _minimum_additive_intersection_rho_and_point(
            normalized_demands, normalized_epsilon,
        )
    requested_normalized_rho = normalized_epsilon * weights[end]
    rho_relation = _intersection_geometry_threshold_relation(
        minimum_normalized_rho, requested_normalized_rho,
    )
    is_touching = rho_relation >= 0
    if rho_relation == 0 &&
       _intersection_geometry_threshold_relation(minimum_normalized_rho, 0.0) == 0
        minimax_distance, minimax_center =
            _minimax_distance_and_center(normalized_demands)
        if _intersection_geometry_threshold_relation(
               minimax_distance, normalized_epsilon,
           ) < 0
            is_touching = false
        end
        feasible_point = minimax_center
    elseif rho_relation > 0
        _multi_item_statistics().additive_radius_repairs += 1
    end
    if is_touching
        _multi_item_statistics().touching_solutions += 1
        touching_demand = number_of_consumers .* feasible_point
        return SO_multi_item_newsvendor_objective_value_and_order(
            0.0,
            [touching_demand],
            [1.0],
            instance_underage_costs,
            instance_overage_costs,
        )
    end

    statistics = _multi_item_statistics()
    active_indices =
        _active_intersection_ball_indices(normalized_demands, normalized_ball_radii)
    statistics.active_ball_count_sum += length(active_indices)
    statistics.total_ball_count_sum += K
    statistics.pruned_solve_observations += 1

    if length(active_indices) == 1
        statistics.single_ball_solutions += 1
        k = active_indices[1]
        return _single_ball_intersection_solution(
            normalized_demands[k],
            normalized_ball_radii[k],
            instance_underage_costs,
            instance_overage_costs,
        )
    end

    active_demands = normalized_demands[active_indices]
    active_radii = normalized_ball_radii[active_indices]
    if multi_item_enable_intersection_dual_solver[]
        dual_solution = _solve_intersection_dual(
            active_demands,
            active_radii,
            feasible_point,
            _initial_intersection_dual_lambda(
                active_demands,
                active_radii .^ 2,
                instance_underage_costs,
                instance_overage_costs,
            ),
            instance_underage_costs,
            instance_overage_costs,
        )
        if !isnothing(dual_solution)
            normalized_objective, normalized_order_values, _ = dual_solution
            candidate_solution = (
                number_of_consumers * normalized_objective,
                number_of_consumers .* normalized_order_values,
            )
            statistics.dual_solver_solutions += 1
            return candidate_solution
        else
            statistics.dual_solver_failures += 1
        end
    end

    problem, normalized_order, lambda, radius_penalty, radius_penalty_constraint =
        _build_intersection_DRO_model(
            active_demands,
            instance_underage_costs,
            instance_overage_costs,
        )
    set_normalized_coefficient(
        fill(radius_penalty_constraint, length(active_indices)),
        collect(lambda),
        -active_radii .^ 2,
    )
    set_objective_coefficient(problem, radius_penalty, 1.0)

    _optimize_multi_item_model!(problem)
    statistics.conic_solutions += 1
    objective = number_of_consumers * objective_value(problem)
    order = number_of_consumers .* value.(normalized_order)
    return objective, order
end


function _multi_item_newsvendor_grid(
    ::typeof(REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order),
    ambiguity_radii,
    demands,
    weight_vectors,
    instance_underage_costs,
    instance_overage_costs,
)
    K = length(demands)
    normalized_demands = [demand ./ number_of_consumers for demand in demands]
    result_type = Tuple{Float64,Vector{Float64}}
    results = Matrix{result_type}(
        undef, length(ambiguity_radii), length(weight_vectors),
    )
    statistics = _multi_item_statistics()

    normalized_epsilons = Vector{Float64}(undef, length(ambiguity_radii))
    for radius_index in eachindex(ambiguity_radii)
        epsilon = ambiguity_radii[radius_index]
        normalized_epsilons[radius_index] = epsilon / number_of_consumers
    end

    radius_ratios = Vector{Float64}(undef, length(weight_vectors))
    for weight_index in eachindex(weight_vectors)
        weights = weight_vectors[weight_index]
        radius_ratios[weight_index] = weights[end]
    end

    # The additive geometry is independent of the radius ratio, so one solve
    # per epsilon is shared by value across all weight columns.
    additive_geometries = Dict{Float64,Tuple{Float64,Vector{Float64}}}()

    problem = nothing
    normalized_order = nothing
    lambda = nothing
    radius_penalty = nothing
    radius_penalty_constraint = nothing
    lambda_is_fixed = falses(K)

    for weight_index in eachindex(weight_vectors)
        radius_ratio = radius_ratios[weight_index]
        relative_radii = [1.0 + (K - k + 1) * radius_ratio for k in 1:K]
        relative_radius_scale = maximum(relative_radii)
        scaled_relative_radii = relative_radii ./ relative_radius_scale
        zero_multiplier_threshold =
            _zero_multiplier_epsilon_threshold(
                normalized_demands,
                relative_radii,
                instance_underage_costs,
                instance_overage_costs,
            )

        radius_penalty_is_current = false
        warm_lambda_full = nothing
        for radius_index in eachindex(ambiguity_radii)
            normalized_epsilon = normalized_epsilons[radius_index]
            normalized_ball_radii = normalized_epsilon .* relative_radii

            if normalized_epsilon >= zero_multiplier_threshold
                statistics.zero_multiplier_solutions += 1
                results[radius_index, weight_index] =
                    _zero_multiplier_solution(
                        instance_underage_costs,
                        instance_overage_costs,
                    )
                continue
            end

            minimum_normalized_rho, additive_feasible_point = get!(
                additive_geometries, normalized_epsilon,
            ) do
                _compute_minimum_additive_intersection_rho_and_point(
                    normalized_demands,
                    normalized_epsilon,
                )
            end
            requested_normalized_rho = normalized_epsilon * radius_ratio
            rho_relation = _intersection_geometry_threshold_relation(
                minimum_normalized_rho, requested_normalized_rho,
            )
            interior_point = additive_feasible_point
            is_touching = rho_relation >= 0
            if rho_relation == 0 &&
               _intersection_geometry_threshold_relation(
                   minimum_normalized_rho, 0.0,
               ) == 0
                minimax_distance, minimax_center =
                    _minimax_distance_and_center(normalized_demands)
                if _intersection_geometry_threshold_relation(
                       minimax_distance, normalized_epsilon,
                   ) < 0
                    is_touching = false
                end
                interior_point = minimax_center
            elseif rho_relation > 0
                statistics.additive_radius_repairs += 1
            end
            if is_touching
                statistics.touching_solutions += 1
                touching_demand = number_of_consumers .* interior_point
                results[radius_index, weight_index] =
                    SO_multi_item_newsvendor_objective_value_and_order(
                        0.0,
                        [touching_demand],
                        [1.0],
                        instance_underage_costs,
                        instance_overage_costs,
                    )
                continue
            end

            active_indices = _active_intersection_ball_indices(
                normalized_demands, normalized_ball_radii,
            )
            statistics.active_ball_count_sum += length(active_indices)
            statistics.total_ball_count_sum += K
            statistics.pruned_solve_observations += 1

            if length(active_indices) == 1
                statistics.single_ball_solutions += 1
                k = active_indices[1]
                results[radius_index, weight_index] =
                    _single_ball_intersection_solution(
                        normalized_demands[k],
                        normalized_ball_radii[k],
                        instance_underage_costs,
                        instance_overage_costs,
                    )
                warm_lambda_full = nothing
                continue
            end

            active_demands = normalized_demands[active_indices]
            active_radii = normalized_ball_radii[active_indices]
            if multi_item_enable_intersection_dual_solver[]
                initial_lambda = isnothing(warm_lambda_full) ?
                    _initial_intersection_dual_lambda(
                        active_demands,
                        active_radii .^ 2,
                        instance_underage_costs,
                        instance_overage_costs,
                    ) :
                    warm_lambda_full[active_indices]
                dual_solution = _solve_intersection_dual(
                    active_demands,
                    active_radii,
                    interior_point,
                    initial_lambda,
                    instance_underage_costs,
                    instance_overage_costs,
                )
                if !isnothing(dual_solution)
                    dual_objective, dual_order, dual_lambda = dual_solution
                    candidate_solution = (
                        number_of_consumers * dual_objective,
                        number_of_consumers .* dual_order,
                    )
                    warm_lambda_full = zeros(K)
                    warm_lambda_full[active_indices] .= dual_lambda
                    statistics.dual_solver_solutions += 1
                    results[radius_index, weight_index] = candidate_solution
                    continue
                else
                    statistics.dual_solver_failures += 1
                end
            end

            if isnothing(problem)
                problem,
                normalized_order,
                lambda,
                radius_penalty,
                radius_penalty_constraint =
                    _build_intersection_DRO_model(
                        normalized_demands,
                        instance_underage_costs,
                        instance_overage_costs,
                    )
            end
            if !radius_penalty_is_current
                set_normalized_coefficient(
                    fill(radius_penalty_constraint, K),
                    collect(lambda),
                    -scaled_relative_radii .^ 2,
                )
                radius_penalty_is_current = true
            end
            is_active = falses(K)
            is_active[active_indices] .= true
            for k in 1:K
                if is_active[k] && lambda_is_fixed[k]
                    unfix(lambda[k])
                    set_lower_bound(lambda[k], 0.0)
                    lambda_is_fixed[k] = false
                elseif !is_active[k] && !lambda_is_fixed[k]
                    fix(lambda[k], 0.0; force = true)
                    lambda_is_fixed[k] = true
                end
            end
            set_objective_coefficient(
                problem,
                radius_penalty,
                (normalized_epsilon * relative_radius_scale)^2,
            )
            _optimize_multi_item_model!(problem)
            statistics.conic_solutions += 1
            results[radius_index, weight_index] = (
                number_of_consumers * objective_value(problem),
                number_of_consumers .* value.(normalized_order),
            )
        end
    end

    return results
end
