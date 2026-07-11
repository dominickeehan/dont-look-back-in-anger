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


# Near-touching intersections can be materially inaccurate even when Gurobi
# reports an optimal solve at a looser tolerance, so retain Gurobi's default
# barrier convergence tolerance for the first attempt.
const multi_item_first_attempt_barrier_gap_tolerance = 1.0e-6
# The closed-form dual solver certifies its own optimality gap against this
# relative tolerance before its solution is accepted over the conic solver.
const multi_item_intersection_dual_relative_gap_tolerance = 1.0e-6
const multi_item_intersection_dual_absolute_gap_tolerance = 1.0e-8
const multi_item_intersection_dual_max_iterations = 2000
const multi_item_enable_intersection_dual_solver = Ref(true)
# A maximizing pair supplies a rigorous lower bound on the intersection
# radius. Accept its equalizer only when its feasible upper bound closes that
# gap to substantially tighter accuracy than the downstream conic solves.
const multi_item_pair_certificate_relative_gap_tolerance = 1.0e-12
const multi_item_pair_certificate_absolute_gap_tolerance = 1.0e-14


Base.@kwdef mutable struct _MultiItemSolverStatistics
    touching_solutions::Int = 0
    zero_multiplier_solutions::Int = 0
    single_ball_solutions::Int = 0
    dual_solver_solutions::Int = 0
    dual_solver_failures::Int = 0
    conic_solutions::Int = 0
    numeric_retry_solves::Int = 0
    geometry_solves::Int = 0
    pair_certificate_solutions::Int = 0
    geometry_socp_solves::Int = 0
    cheap_interior_bound_hits::Int = 0
    active_ball_count_sum::Int = 0
    total_ball_count_sum::Int = 0
    pruned_solve_observations::Int = 0
end


const multi_item_solver_statistics = [
    _MultiItemSolverStatistics() for _ in 1:multi_item_julia_thread_count
]


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
        zero_multiplier_solutions = aggregate.zero_multiplier_solutions,
        single_ball_solutions = aggregate.single_ball_solutions,
        dual_solver_solutions = aggregate.dual_solver_solutions,
        dual_solver_failures = aggregate.dual_solver_failures,
        conic_solutions = aggregate.conic_solutions,
        numeric_retry_solves = aggregate.numeric_retry_solves,
        geometry_solves = aggregate.geometry_solves,
        pair_certificate_solutions = aggregate.pair_certificate_solutions,
        geometry_socp_solves = aggregate.geometry_socp_solves,
        cheap_interior_bound_hits = aggregate.cheap_interior_bound_hits,
        mean_active_balls = mean_active_balls,
        mean_total_balls = mean_total_balls,
    )
end


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
    # Always attempt default numerics first; escalation doubles the solve time,
    # so it is reserved for the (rare) numerically marginal instances.
    set_attribute(problem, "BarHomogeneous", -1)
    set_attribute(problem, "NumericFocus", 0)
    if high_precision
        set_attribute(problem, "FeasibilityTol", 1.0e-9)
        set_attribute(problem, "BarQCPConvTol", 1.0e-10)
    else
        set_attribute(
            problem, "BarQCPConvTol", multi_item_first_attempt_barrier_gap_tolerance,
        )
    end
    optimize!(problem)
    is_solved_and_feasible(problem) && return nothing

    _multi_item_statistics().numeric_retry_solves += 1
    set_attribute(problem, "BarHomogeneous", 1)
    set_attribute(problem, "NumericFocus", 3)
    if !high_precision
        set_attribute(problem, "BarQCPConvTol", 1.0e-6)
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


# Any feasible point of the geometry problem satisfies both scaled-distance
# constraints of a pair, so every pair certifies eps >= ||d_j - d_k|| /
# (relative_j + relative_k); the maximizing pair is also the initial active
# set of the planar solver.
function _pairwise_intersection_epsilon_lower_bound(normalized_demands, relative_radii)
    K = length(normalized_demands)
    best_bound = 0.0
    best_first = 1
    best_second = min(2, K)
    for j in 1:K, k in j+1:K
        bound =
            sqrt(_squared_euclidean_distance(normalized_demands[j], normalized_demands[k])) /
            (relative_radii[j] + relative_radii[k])
        if bound > best_bound
            best_bound = bound
            best_first = j
            best_second = k
        end
    end
    return best_bound, best_first, best_second
end


# The equalizer of a maximizing pair attains the pairwise lower bound. If that
# point lies in the support box and covers every other ball at the same radius,
# it is therefore globally optimal. Floating-point evaluation produces an
# explicit feasible upper bound; returning that upper bound keeps the reported
# point feasible while the tolerance below limits its gap from the rigorous
# pairwise lower bound. Any uncertain case falls through to the exact planar
# solver or the high-precision conic model.
function _maximizing_pair_intersection_certificate(
    normalized_demands,
    relative_radii,
    epsilon_lower_bound,
    first_index,
    second_index,
)
    length(normalized_demands) >= 2 || return nothing
    first_radius = relative_radii[first_index]
    second_radius = relative_radii[second_index]
    radius_sum = first_radius + second_radius
    isfinite(radius_sum) && radius_sum > 0.0 || return nothing

    segment_fraction = first_radius / radius_sum
    feasible_point = Vector{Float64}(undef, multi_item_dimension)
    for i in 1:multi_item_dimension
        first_value = normalized_demands[first_index][i]
        candidate_value = first_value + segment_fraction * (
            normalized_demands[second_index][i] - first_value
        )
        isfinite(candidate_value) && 0.0 <= candidate_value <= 1.0 ||
            return nothing
        feasible_point[i] = candidate_value
    end

    epsilon_upper_bound = 0.0
    for k in eachindex(normalized_demands)
        radius = relative_radii[k]
        isfinite(radius) && radius > 0.0 || return nothing
        required_epsilon =
            sqrt(_squared_euclidean_distance(feasible_point, normalized_demands[k])) /
            radius
        isfinite(required_epsilon) || return nothing
        epsilon_upper_bound = max(epsilon_upper_bound, required_epsilon)
    end

    certificate_tolerance = max(
        multi_item_pair_certificate_absolute_gap_tolerance,
        multi_item_pair_certificate_relative_gap_tolerance * max(
            abs(epsilon_lower_bound), abs(epsilon_upper_bound),
        ),
    )
    epsilon_upper_bound <= epsilon_lower_bound + certificate_tolerance ||
        return nothing
    return max(epsilon_lower_bound, epsilon_upper_bound), feasible_point
end


# Containment pruning for the geometry problem at the pairwise lower bound:
# ||d_j - d_k|| <= (relative_j - relative_k) * tau implies constraint j is
# redundant at every eps >= tau. The pruned optimum still satisfies
# eps >= tau, because replacing a dropped endpoint of the maximizing pair by
# its kept witness only preserves the pairwise bound (triangle inequality),
# so pruning is exact. Every dropped ball has a directly kept witness since
# comparisons only run against already-kept balls.
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


# Equalizer of two scaled distances along the segment between the centers.
function _planar_pair_point(ax, ay, wa, bx, by, wb)
    dx = bx - ax
    dy = by - ay
    fraction = wa / (wa + wb)
    return ax + fraction * dx, ay + fraction * dy
end


# Points where all three scaled distances coincide. Each pairwise locus
# alpha (x.x) - 2 beta.x + gamma = 0 (an Apollonius circle, or a bisector
# line when the weights tie) is combined to eliminate the quadratic term,
# leaving a line that is intersected with one quadratic locus.
function _planar_apollonius_points(ax, ay, wa, bx, by, wb, cx, cy, wc)
    alpha_ab = wb^2 - wa^2
    beta_ab_x = wb^2 * ax - wa^2 * bx
    beta_ab_y = wb^2 * ay - wa^2 * by
    gamma_ab = wb^2 * (ax^2 + ay^2) - wa^2 * (bx^2 + by^2)
    alpha_ac = wc^2 - wa^2
    beta_ac_x = wc^2 * ax - wa^2 * cx
    beta_ac_y = wc^2 * ay - wa^2 * cy
    gamma_ac = wc^2 * (ax^2 + ay^2) - wa^2 * (cx^2 + cy^2)

    if iszero(alpha_ab) && iszero(alpha_ac)
        determinant = 4.0 * (beta_ab_x * beta_ac_y - beta_ab_y * beta_ac_x)
        iszero(determinant) && return 0, NaN, NaN, NaN, NaN
        px = 2.0 * (gamma_ab * beta_ac_y - gamma_ac * beta_ab_y) / determinant
        py = 2.0 * (beta_ab_x * gamma_ac - beta_ac_x * gamma_ab) / determinant
        return 1, px, py, NaN, NaN
    end

    line_x = 2.0 * (alpha_ac * beta_ab_x - alpha_ab * beta_ac_x)
    line_y = 2.0 * (alpha_ac * beta_ab_y - alpha_ab * beta_ac_y)
    line_constant = alpha_ac * gamma_ab - alpha_ab * gamma_ac
    squared_line_norm = line_x^2 + line_y^2
    squared_line_norm > 0.0 || return 0, NaN, NaN, NaN, NaN
    base_x = line_x * line_constant / squared_line_norm
    base_y = line_y * line_constant / squared_line_norm
    inverse_line_norm = 1.0 / sqrt(squared_line_norm)
    tangent_x = -line_y * inverse_line_norm
    tangent_y = line_x * inverse_line_norm

    if abs(alpha_ab) >= abs(alpha_ac)
        alpha, beta_x, beta_y, gamma = alpha_ab, beta_ab_x, beta_ab_y, gamma_ab
    else
        alpha, beta_x, beta_y, gamma = alpha_ac, beta_ac_x, beta_ac_y, gamma_ac
    end
    quadratic_b =
        2.0 * alpha * (base_x * tangent_x + base_y * tangent_y) -
        2.0 * (beta_x * tangent_x + beta_y * tangent_y)
    quadratic_c =
        alpha * (base_x^2 + base_y^2) -
        2.0 * (beta_x * base_x + beta_y * base_y) + gamma
    discriminant = quadratic_b^2 - 4.0 * alpha * quadratic_c
    discriminant >= 0.0 || return 0, NaN, NaN, NaN, NaN
    root = sqrt(discriminant)
    first_step = (-quadratic_b + root) / (2.0 * alpha)
    second_step = (-quadratic_b - root) / (2.0 * alpha)
    return 2,
        base_x + first_step * tangent_x, base_y + first_step * tangent_y,
        base_x + second_step * tangent_x, base_y + second_step * tangent_y
end


# Exact optimum over an active subset by enumerating every candidate point a
# basis of at most three balls can produce (centers, pair equalizers, and
# Apollonius intersections), evaluated after clamping into the support box.
# The true constrained optimum lies in the convex hull of the (in-box)
# centers, hence in the box, and is itself one of the candidates.
function _planar_active_set_optimum(active, coordinates_x, coordinates_y, scaled_radii)
    candidates = Tuple{Float64,Float64}[]
    m = length(active)
    for a_position in 1:m
        a = active[a_position]
        push!(candidates, (coordinates_x[a], coordinates_y[a]))
        for b_position in a_position+1:m
            b = active[b_position]
            push!(
                candidates,
                _planar_pair_point(
                    coordinates_x[a], coordinates_y[a], scaled_radii[a],
                    coordinates_x[b], coordinates_y[b], scaled_radii[b],
                ),
            )
            for c_position in b_position+1:m
                c = active[c_position]
                count, first_x, first_y, second_x, second_y =
                    _planar_apollonius_points(
                        coordinates_x[a], coordinates_y[a], scaled_radii[a],
                        coordinates_x[b], coordinates_y[b], scaled_radii[b],
                        coordinates_x[c], coordinates_y[c], scaled_radii[c],
                    )
                count >= 1 && push!(candidates, (first_x, first_y))
                count >= 2 && push!(candidates, (second_x, second_y))
            end
        end
    end

    best_value = Inf
    best_x = NaN
    best_y = NaN
    for (raw_x, raw_y) in candidates
        point_x = clamp(raw_x, 0.0, 1.0)
        point_y = clamp(raw_y, 0.0, 1.0)
        isfinite(point_x) && isfinite(point_y) || continue
        worst = 0.0
        for k in active
            worst = max(
                worst,
                sqrt(
                    (point_x - coordinates_x[k])^2 +
                    (point_y - coordinates_y[k])^2,
                ) / scaled_radii[k],
            )
        end
        if worst < best_value
            best_value = worst
            best_x = point_x
            best_y = point_y
        end
    end
    return best_value, best_x, best_y
end


# Planar weighted one-center by a growing active set: solve the active
# subset exactly, add the worst violator, and stop once the subset optimum
# covers every ball, which certifies global optimality because subset optima
# never exceed the full optimum. Returns nothing (conic fallback) on any
# degeneracy instead of risking an inexact answer.
function _solve_planar_weighted_one_center(
    normalized_demands, relative_radii, initial_first, initial_second,
)
    K = length(normalized_demands)
    coordinates_x = [demand[1] for demand in normalized_demands]
    coordinates_y = [demand[2] for demand in normalized_demands]
    radius_scale = maximum(relative_radii)
    scaled_radii = relative_radii ./ radius_scale
    all(radius -> isfinite(radius) && radius^2 > 0.0, scaled_radii) ||
        return nothing

    if K == 1
        return 0.0, [
            clamp(coordinates_x[1], 0.0, 1.0),
            clamp(coordinates_y[1], 0.0, 1.0),
        ]
    end

    active = [initial_first, initial_second]
    for _ in 1:12
        value, point_x, point_y = _planar_active_set_optimum(
            active, coordinates_x, coordinates_y, scaled_radii,
        )
        isfinite(value) || return nothing
        certificate_threshold = value * (1.0 + 1.0e-12) + 1.0e-15
        worst_index = 0
        worst_value = certificate_threshold
        for k in 1:K
            coverage = sqrt(
                (point_x - coordinates_x[k])^2 +
                (point_y - coordinates_y[k])^2,
            ) / scaled_radii[k]
            if coverage > worst_value
                worst_value = coverage
                worst_index = k
            end
        end
        worst_index == 0 && return value / radius_scale, [point_x, point_y]
        worst_index in active && return nothing
        push!(active, worst_index)
    end
    return nothing
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

    _multi_item_statistics().geometry_solves += 1
    epsilon_lower_bound, initial_first, initial_second =
        _pairwise_intersection_epsilon_lower_bound(normalized_demands, relative_radii)
    pair_certificate = _maximizing_pair_intersection_certificate(
        normalized_demands,
        relative_radii,
        epsilon_lower_bound,
        initial_first,
        initial_second,
    )
    if !isnothing(pair_certificate)
        _multi_item_statistics().pair_certificate_solutions += 1
        return pair_certificate
    end
    constraining_indices = _geometry_constraining_ball_indices(
        normalized_demands, relative_radii, epsilon_lower_bound,
    )
    constraining_demands = normalized_demands[constraining_indices]
    constraining_radii = relative_radii[constraining_indices]

    if multi_item_dimension == 2 &&
       all(
           demand -> all(value -> 0.0 <= value <= 1.0, demand),
           constraining_demands,
       )
        _, initial_first, initial_second = _pairwise_intersection_epsilon_lower_bound(
            constraining_demands, constraining_radii,
        )
        planar_solution = _solve_planar_weighted_one_center(
            constraining_demands, constraining_radii, initial_first, initial_second,
        )
        isnothing(planar_solution) || return planar_solution
    end

    _multi_item_statistics().geometry_socp_solves += 1
    geometry_problem = _new_multi_item_model()
    @variables(geometry_problem, begin
        1.0 >= feasible_point[i = 1:multi_item_dimension] >= 0.0
        minimum_normalized_epsilon >= 0.0
    end)
    for k in eachindex(constraining_indices)
        @constraint(
            geometry_problem,
            [
                constraining_radii[k] * minimum_normalized_epsilon;
                [
                    feasible_point[i] - constraining_demands[k][i]
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


function _squared_euclidean_distance(first_point, second_point)
    total = 0.0
    for i in eachindex(first_point)
        difference = first_point[i] - second_point[i]
        total += difference * difference
    end
    return total
end


# W2(P, delta_d)^2 = E||X - d||^2 <= sum_i max(|d_i|, |d_i - 1|)^2 for every
# distribution P supported on [0, 1]^n, so a ball at least that large in
# radius constrains nothing and can be dropped exactly.
function _intersection_ball_is_vacuous(normalized_demand, normalized_radius)
    farthest_squared_distance = 0.0
    for value in normalized_demand
        farthest = max(abs(value), abs(value - 1.0))
        farthest_squared_distance += farthest * farthest
    end
    return normalized_radius * normalized_radius >= farthest_squared_distance
end


# Exact pruning of the intersection: by the W2 triangle inequality, ball j
# contains ball k whenever ||d_j - d_k||_2 <= r_j - r_k, making constraint j
# redundant. Processing balls by ascending radius and comparing only against
# already-kept balls suffices, because a dropped comparator is itself implied
# by an earlier kept ball. Exact duplicates keep their first occurrence.
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


# The zero-multiplier condition radius_k^2 >= required_k^2 scales as
# epsilon * relative_radius_k, so it holds if and only if the normalized
# epsilon reaches this threshold; computing it once replaces per-epsilon
# checks across a radius grid.
function _zero_multiplier_epsilon_threshold(normalized_demands, relative_radii)
    critical_fractile =
        multi_item_underage_cost /
        (multi_item_overage_cost + multi_item_underage_cost)
    threshold = 0.0
    for k in eachindex(normalized_demands)
        required_squared_radius = 0.0
        for value in normalized_demands[k]
            required_squared_radius +=
                (1.0 - critical_fractile) * (1.0 - value)^2 +
                critical_fractile * value^2
        end
        threshold = max(threshold, sqrt(required_squared_radius) / relative_radii[k])
    end
    return threshold
end


# Cheap upper bound on the minimum feasible intersection epsilon: evaluate the
# max scaled distance at each (box-clamped) demand point and at their mean.
# Whenever the requested epsilon exceeds this bound, the intersection is known
# to be nonempty without solving the geometry model, and the minimizing
# candidate is a strictly interior point reused by the dual certificate.
function _minimum_intersection_epsilon_upper_bound_and_point(
    normalized_demands, relative_radii,
)
    K = length(normalized_demands)
    mean_point = zeros(multi_item_dimension)
    for demand in normalized_demands
        mean_point .+= demand
    end
    mean_point ./= K
    clamp!(mean_point, 0.0, 1.0)

    best_epsilon = Inf
    best_point = mean_point
    candidate = similar(mean_point)
    for candidate_index in 0:K
        if candidate_index == 0
            candidate .= mean_point
        else
            candidate .= normalized_demands[candidate_index]
            clamp!(candidate, 0.0, 1.0)
        end
        worst_scaled_distance = 0.0
        for k in 1:K
            worst_scaled_distance = max(
                worst_scaled_distance,
                sqrt(_squared_euclidean_distance(candidate, normalized_demands[k])) /
                relative_radii[k],
            )
        end
        if worst_scaled_distance < best_epsilon
            best_epsilon = worst_scaled_distance
            best_point = copy(candidate)
        end
    end
    return best_epsilon, best_point
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


function _zero_multiplier_solution(doubling_count)
    critical_fractile =
        multi_item_underage_cost /
        (multi_item_overage_cost + multi_item_underage_cost)
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
    return _zero_multiplier_solution(doubling_count)
end


# A single remaining ball is an ordinary W2 ambiguity set around a point mass,
# which the weighted-W2 machinery (closed form when the demand lies in the
# support box, conic otherwise) already solves.
function _single_ball_intersection_solution(
    normalized_demand, normalized_radius, doubling_count,
)
    demand = multi_item_demand_upper_bound .* normalized_demand
    epsilon = multi_item_demand_upper_bound * normalized_radius
    return W2_DRO_multi_item_newsvendor_objective_value_and_order(
        epsilon, [demand], [1.0], doubling_count,
    )
end


# For multipliers lambda >= 0, one per ball, weak duality gives the bound
#   OPT <= G(lambda) = sum_k lambda_k r_k^2
#          + sum_i min_{x_i in [0,1]} max_l sup_{xi in [0,1]}
#            [piece_l(x_i, xi) - sum_k lambda_k (xi - d_ki)^2],
# because W2(P, delta_d)^2 = E||X - d||^2 exactly. The inner sup is the
# bounded linear-quadratic conjugate at the weighted center c_i with mass
# Lambda = sum_k lambda_k, so G and its gradient (via Danskin's theorem:
# dG/dlambda_k = r_k^2 - sum_{i,l} w_il (xi*_il - d_ki)^2) are closed form.
# This evaluates G, the gradient, and the per-item worst-case atoms
# xi*_U >= xi*_O with piece weights (w_U, w_O) and the minimizing order.
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
)
    K, n = size(demand_matrix)
    underage = multi_item_underage_cost
    overage = multi_item_overage_cost
    total_multiplier = 0.0
    objective = 0.0
    for k in 1:K
        total_multiplier += lambda[k]
        objective += lambda[k] * squared_radii[k]
        gradient[k] = squared_radii[k]
    end

    for i in 1:n
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
            if overage > 0.0
                lower_displacement = min(center, overage / (2.0 * total_multiplier))
                overage_conjugate =
                    -overage * (center - lower_displacement) -
                    total_multiplier * lower_displacement^2
                atom_lower[i] = center - lower_displacement
            else
                overage_conjugate = 0.0
                atom_lower[i] = center
            end
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


# Rigorous acceptance test for a candidate dual point. G(lambda) is a valid
# upper bound on OPT for every lambda >= 0. The per-item two-atom worst-case
# distribution P gives min_x E_P[loss], a valid lower bound, once P is made
# feasible: mixing with the point mass at a strictly interior point z of the
# intersection, P' = (1 - theta) P + theta delta_z, repairs a per-ball moment
# violation v_k at contraction theta >= v_k / (r_k^2 - ||z - d_k||^2), since
# the squared W2 distance to a point mass is linear in the distribution. The
# solution is accepted only if G - (1 - theta) LB is within tolerance.
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
)
    isfinite(objective) || return nothing
    K, n = size(demand_matrix)
    underage = multi_item_underage_cost
    overage = multi_item_overage_cost

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
            min(weight_upper[i] * underage, weight_lower[i] * overage)
    end
    lower_bound *= 1.0 - contraction

    gap = objective - lower_bound
    tolerance =
        multi_item_intersection_dual_absolute_gap_tolerance +
        multi_item_intersection_dual_relative_gap_tolerance * abs(objective)
    gap <= tolerance || return nothing
    return objective, copy(order)
end


function _initial_intersection_dual_lambda(active_normalized_demands, squared_radii)
    lambda = zeros(length(squared_radii))
    smallest_index = argmin(squared_radii)
    closed_form_data = _prepare_weighted_W2_closed_form(
        [multi_item_demand_upper_bound .* active_normalized_demands[smallest_index]],
        [1.0],
    )
    isnothing(closed_form_data) && return lambda
    _, _, displacement_terms = closed_form_data
    lambda[smallest_index] =
        _optimal_weighted_W2_lambda(squared_radii[smallest_index], displacement_terms)
    return lambda
end


# Minimizes the convex, essentially C^1 dual G over lambda >= 0 with a
# projected Barzilai-Borwein gradient method (nonmonotone Armijo safeguard).
# Every candidate must pass the rigorous certificate above before being
# returned, so failure to converge simply falls back to the conic solver.
function _solve_intersection_dual(
    active_normalized_demands,
    active_normalized_ball_radii,
    interior_point,
    initial_lambda,
)
    K = length(active_normalized_demands)
    n = multi_item_dimension
    demand_matrix = Matrix{Float64}(undef, K, n)
    for k in 1:K, i in 1:n
        demand_matrix[k, i] = active_normalized_demands[k][i]
    end
    squared_radii = active_normalized_ball_radii .^ 2
    all(isfinite, squared_radii) || return nothing

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
        lambda, demand_matrix, squared_radii,
    )
    isfinite(objective) || return nothing

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
                trial_lambda, demand_matrix, squared_radii,
            )
            if isfinite(trial_objective) &&
               trial_objective <=
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
    )
    isnothing(certified) && return nothing
    certified_objective, certified_order = certified
    return certified_objective, certified_order, copy(lambda)
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
    if !isnothing(zero_multiplier_solution)
        _multi_item_statistics().zero_multiplier_solutions += 1
        return zero_multiplier_solution
    end

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
        _multi_item_statistics().touching_solutions += 1
        touching_demand = multi_item_demand_upper_bound .* feasible_point
        return SO_multi_item_newsvendor_objective_value_and_order(
            0.0, [touching_demand], [1.0], doubling_count,
        )
    end

    statistics = _multi_item_statistics()
    active_indices =
        _active_intersection_ball_indices(normalized_demands, normalized_ball_radii)
    statistics.active_ball_count_sum += length(active_indices)
    statistics.total_ball_count_sum += K
    statistics.pruned_solve_observations += 1

    if isempty(active_indices)
        # Unreachable in exact arithmetic: an all-vacuous intersection implies
        # the zero-multiplier condition already returned above.
        statistics.zero_multiplier_solutions += 1
        return _zero_multiplier_solution(doubling_count)
    end
    if length(active_indices) == 1
        statistics.single_ball_solutions += 1
        k = active_indices[1]
        return _single_ball_intersection_solution(
            normalized_demands[k], normalized_ball_radii[k], doubling_count,
        )
    end

    active_demands = normalized_demands[active_indices]
    active_radii = normalized_ball_radii[active_indices]
    if multi_item_enable_intersection_dual_solver[] &&
       all(demand -> all(value -> 0.0 <= value <= 1.0, demand), active_demands)
        dual_solution = _solve_intersection_dual(
            active_demands,
            active_radii,
            feasible_point,
            _initial_intersection_dual_lambda(active_demands, active_radii .^ 2),
        )
        if !isnothing(dual_solution)
            statistics.dual_solver_solutions += 1
            normalized_objective, normalized_order_values, _ = dual_solution
            return (
                multi_item_demand_upper_bound * normalized_objective,
                multi_item_demand_upper_bound .* normalized_order_values,
                doubling_count,
            )
        end
        statistics.dual_solver_failures += 1
    end

    problem, normalized_order, lambda, radius_penalty, radius_penalty_constraint =
        _build_intersection_DRO_model(active_demands)
    set_normalized_coefficient(
        fill(radius_penalty_constraint, length(active_indices)),
        collect(lambda),
        -active_radii .^ 2,
    )
    set_objective_coefficient(problem, radius_penalty, 1.0)

    _optimize_multi_item_model!(problem)
    statistics.conic_solutions += 1
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
    statistics = _multi_item_statistics()

    problem = nothing
    normalized_order = nothing
    lambda = nothing
    radius_penalty = nothing
    radius_penalty_constraint = nothing
    lambda_is_fixed = falses(K)

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

        # Both regime boundaries are monotone in epsilon for a fixed radius
        # ratio, so they are decided by thresholds computed once per weight
        # vector, and the geometry model is only solved when some epsilon
        # falls below the cheap interior upper bound.
        zero_multiplier_threshold =
            _zero_multiplier_epsilon_threshold(normalized_demands, relative_radii)
        interior_bound_epsilon, interior_bound_point =
            _minimum_intersection_epsilon_upper_bound_and_point(
                normalized_demands, relative_radii,
            )

        geometry = nothing
        radius_penalty_is_current = false
        warm_lambda_full = nothing
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

            if normalized_epsilon >= zero_multiplier_threshold
                statistics.zero_multiplier_solutions += 1
                results[radius_index, weight_index] =
                    _zero_multiplier_solution(doubling_count)
                continue
            end

            interior_point = nothing
            if !isnothing(geometry) ||
               normalized_epsilon <=
               interior_bound_epsilon * (1.0 + 1.0e-12) + 1.0e-15
                if isnothing(geometry)
                    geometry = _compute_minimum_intersection_epsilon_and_point(
                        normalized_demands, radius_ratio,
                    )
                end
                minimum_normalized_epsilon, feasible_point = geometry
                if minimum_normalized_epsilon >= normalized_epsilon
                    statistics.touching_solutions += 1
                    touching_demand = multi_item_demand_upper_bound .* feasible_point
                    results[radius_index, weight_index] =
                        SO_multi_item_newsvendor_objective_value_and_order(
                            0.0, [touching_demand], [1.0], doubling_count,
                        )
                    continue
                end
                interior_point = feasible_point
            else
                statistics.cheap_interior_bound_hits += 1
                interior_point = interior_bound_point
            end

            active_indices = _active_intersection_ball_indices(
                normalized_demands, normalized_ball_radii,
            )
            statistics.active_ball_count_sum += length(active_indices)
            statistics.total_ball_count_sum += K
            statistics.pruned_solve_observations += 1

            if isempty(active_indices)
                # Unreachable in exact arithmetic: an all-vacuous intersection
                # implies the zero-multiplier threshold already fired above.
                statistics.zero_multiplier_solutions += 1
                results[radius_index, weight_index] =
                    _zero_multiplier_solution(doubling_count)
                continue
            end
            if length(active_indices) == 1
                statistics.single_ball_solutions += 1
                k = active_indices[1]
                results[radius_index, weight_index] =
                    _single_ball_intersection_solution(
                        normalized_demands[k],
                        normalized_ball_radii[k],
                        doubling_count,
                    )
                warm_lambda_full = nothing
                continue
            end

            active_demands = normalized_demands[active_indices]
            active_radii = normalized_ball_radii[active_indices]
            if multi_item_enable_intersection_dual_solver[] &&
               all(demand -> all(value -> 0.0 <= value <= 1.0, demand), active_demands)
                initial_lambda = isnothing(warm_lambda_full) ?
                    _initial_intersection_dual_lambda(
                        active_demands, active_radii .^ 2,
                    ) :
                    warm_lambda_full[active_indices]
                dual_solution = _solve_intersection_dual(
                    active_demands, active_radii, interior_point, initial_lambda,
                )
                if !isnothing(dual_solution)
                    statistics.dual_solver_solutions += 1
                    dual_objective, dual_order, dual_lambda = dual_solution
                    results[radius_index, weight_index] = (
                        multi_item_demand_upper_bound * dual_objective,
                        multi_item_demand_upper_bound .* dual_order,
                        doubling_count,
                    )
                    warm_lambda_full = zeros(K)
                    warm_lambda_full[active_indices] .= dual_lambda
                    continue
                end
                statistics.dual_solver_failures += 1
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
            # Pruned balls keep their columns in the reused model with their
            # multipliers fixed at zero, which presolve removes cheaply.
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
                multi_item_demand_upper_bound * objective_value(problem),
                multi_item_demand_upper_bound .* value.(normalized_order),
                doubling_count,
            )
        end
    end

    return results
end
