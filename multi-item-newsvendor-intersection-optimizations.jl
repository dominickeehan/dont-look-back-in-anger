# Intersection-specific geometry, solver shortcuts, and grid execution.
# This file is included at the end of multi-item-newsvendor-optimizations.jl
# and uses the shared model, SAA, and weighted-W2 functions defined there.

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


# If one ball contains another, the larger ball is redundant. A shared radius
# increase preserves this containment relation.
function _geometry_constraining_ball_indices(
    normalized_demands, normalized_base_radii,
)
    permutation = sortperm(normalized_base_radii)
    kept = Int[]
    for j in permutation
        contained = false
        for k in kept
            radius_gap = normalized_base_radii[j] - normalized_base_radii[k]
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


# The repair enlarges every ball by one shared increase a, so the smallest
# feasible a at a candidate point is the largest amount by which that point
# overshoots any ball. A negative value means the point is strictly inside the
# intersection and the balls could even be shrunk.
function _required_radius_increase_at_point(
    point, normalized_demands, normalized_base_radii,
)
    required_increase = -Inf
    for k in eachindex(normalized_demands)
        distance = sqrt(_squared_euclidean_distance(point, normalized_demands[k]))
        required_increase = max(
            required_increase, distance - normalized_base_radii[k],
        )
    end
    return required_increase
end


function _radius_increase_lower_bound_and_candidate(
    normalized_demands, normalized_base_radii,
)
    K = length(normalized_demands)
    lower_bound = -Inf
    first_index = 0
    second_index = 0

    # The contact point stays inside the support box, so no ball can be reached
    # more cheaply than from its projection onto the box.
    for k in 1:K
        projected = clamp.(normalized_demands[k], 0.0, 1.0)
        support_distance = sqrt(
            _squared_euclidean_distance(normalized_demands[k], projected),
        )
        bound = support_distance - normalized_base_radii[k]
        if bound > lower_bound
            lower_bound = bound
            first_index = k
            second_index = 0
        end
    end

    # Two balls separated by D need a >= (D - r_j - r_k) / 2, since the shared
    # increase has to close the gap from both sides.
    for j in 1:K, k in j+1:K
        center_distance = sqrt(
            _squared_euclidean_distance(normalized_demands[j], normalized_demands[k]),
        )
        bound =
            (
                center_distance -
                normalized_base_radii[j] -
                normalized_base_radii[k]
            ) / 2.0
        if bound > lower_bound
            lower_bound = bound
            first_index = j
            second_index = k
        end
    end
    return lower_bound, first_index, second_index
end


function _radius_increase_candidate_certificate(
    normalized_demands,
    normalized_base_radii,
    increase_lower_bound,
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
            normalized_base_radii[first_index] + increase_lower_bound
        fraction = first_radius / center_distance
        0.0 <= fraction <= 1.0 || return nothing
        normalized_demands[first_index] .+ fraction .* (
            normalized_demands[second_index] .- normalized_demands[first_index]
        )
    end
    all(value -> 0.0 <= value <= 1.0, candidate) || return nothing

    increase_upper_bound = _required_radius_increase_at_point(
        candidate, normalized_demands, normalized_base_radii,
    )
    certificate_tolerance = max(
        multi_item_pair_certificate_absolute_gap_tolerance,
        multi_item_pair_certificate_relative_gap_tolerance * max(
            abs(increase_lower_bound), abs(increase_upper_bound),
        ),
    )
    increase_upper_bound <= increase_lower_bound + certificate_tolerance ||
        return nothing
    return max(increase_lower_bound, increase_upper_bound), candidate
end


function _radius_increase_candidate_upper_bound_and_point(
    normalized_demands, normalized_base_radii,
)
    K = length(normalized_demands)
    mean_point = zeros(number_of_items)
    for demand in normalized_demands
        mean_point .+= demand
    end
    mean_point ./= K
    clamp!(mean_point, 0.0, 1.0)

    best_increase = Inf
    best_point = mean_point
    candidate = similar(mean_point)
    for candidate_index in 0:K
        if candidate_index == 0
            candidate .= mean_point
        else
            candidate .= normalized_demands[candidate_index]
            clamp!(candidate, 0.0, 1.0)
        end
        required_increase = _required_radius_increase_at_point(
            candidate, normalized_demands, normalized_base_radii,
        )
        if required_increase < best_increase
            best_increase = required_increase
            best_point = copy(candidate)
        end
    end
    return best_increase, best_point
end


function _certified_radius_increase_and_lower_bound(
    normalized_demands, normalized_base_radii,
)
    increase_lower_bound, first_index, second_index =
        _radius_increase_lower_bound_and_candidate(
            normalized_demands, normalized_base_radii,
        )
    certificate = _radius_increase_candidate_certificate(
        normalized_demands,
        normalized_base_radii,
        increase_lower_bound,
        first_index,
        second_index,
    )
    !isnothing(certificate) && return certificate, increase_lower_bound

    candidate_upper_bound, candidate_point =
        _radius_increase_candidate_upper_bound_and_point(
            normalized_demands, normalized_base_radii,
        )
    candidate_tolerance = max(
        multi_item_pair_certificate_absolute_gap_tolerance,
        multi_item_pair_certificate_relative_gap_tolerance * max(
            abs(increase_lower_bound), abs(candidate_upper_bound),
        ),
    )
    if candidate_upper_bound <= increase_lower_bound + candidate_tolerance
        return (
            max(increase_lower_bound, candidate_upper_bound), candidate_point,
        ), increase_lower_bound
    end
    return nothing, increase_lower_bound
end


struct _RadiusIncreaseModel
    problem::Model
    point::Vector{VariableRef}
    increase::VariableRef
    ball_constraints::Vector{ConstraintRef}
    constraining_indices::Vector{Int}
end


function _build_radius_increase_model(normalized_demands, constraining_indices)
    problem = _new_multi_item_model()
    @variables(problem, begin
        1.0 >= point[i = 1:number_of_items] >= 0.0
        increase
    end)
    ball_constraints = Vector{ConstraintRef}(undef, length(constraining_indices))
    for (position, k) in enumerate(constraining_indices)
        ball_constraints[position] = @constraint(
            problem,
            [
                increase;
                [point[i] - normalized_demands[k][i] for i in 1:number_of_items]
            ] in MathOptInterface.SecondOrderCone(number_of_items + 1),
        )
    end
    @objective(problem, Min, increase)
    return _RadiusIncreaseModel(
        problem, point, increase, ball_constraints, constraining_indices,
    )
end


function _solve_radius_increase_model!(
    model_data,
    normalized_demands,
    normalized_base_radii,
    increase_lower_bound,
)
    for (position, k) in enumerate(model_data.constraining_indices)
        constants = [
            normalized_base_radii[k];
            [-value for value in normalized_demands[k]]
        ]
        MathOptInterface.modify(
            backend(model_data.problem),
            index(model_data.ball_constraints[position]),
            MathOptInterface.VectorConstantChange(constants),
        )
    end
    _optimize_multi_item_model!(model_data.problem; high_precision = true)

    point_value = clamp.(value.(model_data.point), 0.0, 1.0)
    required_increase = _required_radius_increase_at_point(
        point_value, normalized_demands, normalized_base_radii,
    )
    return max(
        increase_lower_bound,
        value(model_data.increase),
        required_increase,
    ), point_value
end


function _compute_minimum_radius_increase_and_point(
    normalized_demands, normalized_base_radii,
)
    certificate, increase_lower_bound =
        _certified_radius_increase_and_lower_bound(
            normalized_demands, normalized_base_radii,
        )
    !isnothing(certificate) && return certificate
    constraining_indices = _geometry_constraining_ball_indices(
        normalized_demands, normalized_base_radii,
    )
    return _solve_radius_increase_model!(
        _build_radius_increase_model(normalized_demands, constraining_indices),
        normalized_demands,
        normalized_base_radii,
        increase_lower_bound,
    )
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
        return zero_multiplier_solution
    end

    # Holding epsilon and rho fixed, the intersection is nonempty exactly when
    # the smallest shared radius increase is negative. At zero the balls touch,
    # and above zero the set is empty and every radius grows by that increase;
    # in both cases the ambiguity set collapses to the point mass at the
    # first-contact point.
    minimum_normalized_increase, feasible_point =
        _compute_minimum_radius_increase_and_point(
            normalized_demands, normalized_ball_radii,
        )
    increase_relation = _intersection_geometry_threshold_relation(
        minimum_normalized_increase, 0.0,
    )
    is_touching = increase_relation >= 0
    if is_touching
        touching_demand = number_of_consumers .* feasible_point
        return SO_multi_item_newsvendor_objective_value_and_order(
            0.0,
            [touching_demand],
            [1.0],
            instance_underage_costs,
            instance_overage_costs,
        )
    end

    active_indices =
        _active_intersection_ball_indices(normalized_demands, normalized_ball_radii)

    if length(active_indices) == 1
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
            return candidate_solution
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

    geometry_models = Dict{Tuple{Vararg{Int}},_RadiusIncreaseModel}()
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
                results[radius_index, weight_index] =
                    _zero_multiplier_solution(
                        instance_underage_costs,
                        instance_overage_costs,
                    )
                continue
            end

            certificate, increase_lower_bound =
                _certified_radius_increase_and_lower_bound(
                    normalized_demands, normalized_ball_radii,
                )
            if isnothing(certificate)
                constraining_indices = _geometry_constraining_ball_indices(
                    normalized_demands, normalized_ball_radii,
                )
                geometry_model_data = get!(
                    geometry_models, Tuple(constraining_indices),
                ) do
                    _build_radius_increase_model(
                        normalized_demands, constraining_indices,
                    )
                end
                minimum_normalized_increase, interior_point =
                    _solve_radius_increase_model!(
                        geometry_model_data,
                        normalized_demands,
                        normalized_ball_radii,
                        increase_lower_bound,
                    )
            else
                minimum_normalized_increase, interior_point = certificate
            end
            increase_relation = _intersection_geometry_threshold_relation(
                minimum_normalized_increase, 0.0,
            )
            is_touching = increase_relation >= 0
            if is_touching
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

            if length(active_indices) == 1
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
                    results[radius_index, weight_index] = candidate_solution
                    continue
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
            results[radius_index, weight_index] = (
                number_of_consumers * objective_value(problem),
                number_of_consumers .* value.(normalized_order),
            )
        end
    end

    return results
end
