using LinearAlgebra, Random


# First-order dual solver for an intersection of Wasserstein balls, replacing
# the exact conic reformulation in
# multi-item-newsvendor-intersection-conic-optimizations.jl.


const multi_item_geometry_tolerance = 1.0e-6


function _euclidean_distance(first_point, second_point)
    squared_distance = 0.0
    @inbounds for i in eachindex(first_point, second_point)
        squared_distance += (first_point[i] - second_point[i])^2
    end
    return sqrt(squared_distance)
end


# In the following ball-intersection problem it is enough to work with points
# ξ in R^m because
# W₂(P, 1_ξ) = sqrt(sum((E(P)_i - ξ_i)^2) + Tr(Cov(P))). At first
# contact the covariance is therefore zero and P is a point mass.


function _pairwise_distances(demands)
    K = length(demands)
    distances = zeros(K, K)
    for j in 1:K, k in j+1:K
        distances[j, k] = distances[k, j] =
            _euclidean_distance(demands[j], demands[k])
    end
    return distances
end


# If ball j contains ball k, ball j is redundant in their intersection. A
# shared additive radius increase preserves the containment relation exactly.
function _nonredundant_ball_indices(ball_radii, pair_distances)
    kept_indices = Int[]
    # REMK_intersection_ball_radii constructs the radii in nonincreasing order.
    for j in Iterators.reverse(eachindex(ball_radii))
        contained_ball_already_kept = any(
            pair_distances[j, k] + ball_radii[k] <= ball_radii[j]
            for k in kept_indices
        )
        contained_ball_already_kept || push!(kept_indices, j)
    end
    return sort!(kept_indices)
end


# The shared radius increase required at a candidate point. A negative value
# directly certifies a nonempty interior and hence Slater's condition.
function _required_radius_increase_at_point(point, demands, ball_radii)
    return maximum(
        _euclidean_distance(point, demands[k]) - ball_radii[k]
        for k in eachindex(demands)
    )
end


# Every pair supplies the lower bound
#   a >= (‖d_j - d_k‖ - r_j - r_k) / 2.
# The smallest radius also supplies a >= -min(r). If the contact point for the
# largest bound lies strictly inside every ball, it certifies nonemptiness. If
# its feasible upper bound matches the lower bound, it is globally optimal. The
# critical indices are also returned as an ordering hint for the exact fallback.
function _certified_two_ball_radius_increase(
    demands, ball_radii, pair_distances,
)
    K = length(demands)
    lower_bound = -minimum(ball_radii)
    critical_j = 0
    critical_k = argmin(ball_radii)
    for j in 1:K, k in j+1:K
        bound =
            (pair_distances[j, k] - ball_radii[j] - ball_radii[k]) / 2.0
        if bound > lower_bound
            lower_bound = bound
            critical_j, critical_k = j, k
        end
    end
    critical_indices =
        critical_j == 0 ? [critical_k] : [critical_j, critical_k]

    candidate = if critical_j == 0
        demands[critical_k]
    else
        distance = pair_distances[critical_j, critical_k]
        distance > 0.0 || return nothing, critical_indices
        fraction = (ball_radii[critical_j] + lower_bound) / distance
        0.0 <= fraction <= 1.0 || return nothing, critical_indices
        demands[critical_j] +
            fraction * (demands[critical_k] - demands[critical_j])
    end
    candidate_increase = _required_radius_increase_at_point(
        candidate, demands, ball_radii,
    )
    tolerance = 1.0e-6 * max(
        1.0, abs(lower_bound), abs(candidate_increase),
    )
    # Any strictly interior point is enough to justify the dual, so this branch
    # does not need the exact minimum radius increase.
    candidate_increase < -tolerance &&
        return (candidate_increase, candidate), critical_indices
    candidate_increase <= lower_bound + tolerance ||
        return nothing, critical_indices
    return (max(lower_bound, candidate_increase), candidate), critical_indices
end


# The ball supported by the listed active constraints. The center lies in the
# affine hull of their centers, so subtracting the first tangency equation from
# the others leaves a small linear system and one quadratic in the increase.
function _ball_from_support(demands, ball_radii, support)
    first_index = first(support)
    first_demand = demands[first_index]
    first_radius = ball_radii[first_index]
    length(support) == 1 && return -first_radius, copy(first_demand)

    other_indices = support[2:end]
    differences = Matrix{Float64}(
        undef, length(first_demand), length(other_indices),
    )
    for (column, index) in enumerate(other_indices)
        differences[:, column] = demands[index] - first_demand
    end
    gram = differences' * differences
    radius_differences = ball_radii[other_indices] .- first_radius
    right_hand_sides = hcat(
        0.5 .* (
            diag(gram) .-
            (ball_radii[other_indices] .^ 2 .- first_radius^2)
        ),
        -radius_differences,
    )
    # The Gram matrix is nonsingular exactly when these center differences are
    # independent. One factorization therefore replaces a separate rank SVD and
    # solves both unchanged linear systems.
    factorization = cholesky(Symmetric(gram); check = false)
    issuccess(factorization) || return nothing
    coordinates = factorization \ right_hand_sides
    constant_coordinates = coordinates[:, 1]
    increase_coordinates = coordinates[:, 2]
    constant_offset = differences * constant_coordinates
    increase_offset = differences * increase_coordinates

    quadratic = dot(increase_offset, increase_offset) - 1.0
    linear = 2.0 * (
        dot(constant_offset, increase_offset) - first_radius
    )
    constant = dot(constant_offset, constant_offset) - first_radius^2
    roots = if abs(quadratic) <= 1.0e-6
        abs(linear) <= 1.0e-6 && return nothing
        (-constant / linear,)
    else
        discriminant = linear^2 - 4.0 * quadratic * constant
        discriminant >= -1.0e-6 || return nothing
        root_offset = sqrt(max(0.0, discriminant))
        (
            (-linear - root_offset) / (2.0 * quadratic),
            (-linear + root_offset) / (2.0 * quadratic),
        )
    end

    valid_roots = [
        root for root in roots if all(
            root + ball_radii[index] >= -1.0e-6 for index in support
        )
    ]
    isempty(valid_roots) && return nothing
    increase = minimum(valid_roots)
    point = first_demand + constant_offset + increase * increase_offset
    return increase, point
end


# The recursive active set has at most number_of_items + 1 balls. Trying its
# subsets also covers dependent or nonminimal supports without special cases.
function _smallest_supported_ball(demands, ball_radii, support)
    isempty(support) && return -Inf, zeros(length(first(demands)))
    best_geometry = (Inf, zeros(length(first(demands))))
    for mask in 1:(1 << length(support)) - 1
        subset = [
            support[j] for j in eachindex(support)
            if mask & (1 << (j - 1)) != 0
        ]
        geometry = _ball_from_support(demands, ball_radii, subset)
        isnothing(geometry) && continue
        increase, point = geometry
        increase < best_geometry[1] || continue
        all(
            _euclidean_distance(point, demands[index]) <=
                increase + ball_radii[index] + 1.0e-6
            for index in support
        ) && (best_geometry = geometry)
    end
    return best_geometry
end


function _active_set_radius_increase(
    demands, ball_radii, indices, support, count,
)
    if count == 0 || length(support) == length(first(demands)) + 1
        return _smallest_supported_ball(demands, ball_radii, support)
    end

    geometry = _active_set_radius_increase(
        demands, ball_radii, indices, support, count - 1,
    )
    increase, point = geometry
    index = indices[count]
    _euclidean_distance(point, demands[index]) <=
        increase + ball_radii[index] + 1.0e-6 && return geometry

    push!(support, index)
    geometry = _active_set_radius_increase(
        demands, ball_radii, indices, support, count - 1,
    )
    pop!(support)
    return geometry
end


# Preferred balls are processed first to find a strong partial solution sooner.
# They are not forced into the support, and every ball still passes through the
# same recursion, so the order changes the work but not the exact answer.
function _active_set_radius_increase(
    demands, ball_radii, preferred_indices = Int[],
)
    indices = shuffle!(Xoshiro(1), collect(eachindex(demands)))
    indices = vcat(
        preferred_indices,
        filter(index -> index ∉ preferred_indices, indices),
    )
    return _active_set_radius_increase(
        demands, ball_radii, indices, Int[], length(demands),
    )
end


# A type-2 Wasserstein ball around the point mass at demands[k] is the moment
# constraint E_P ‖ξ - demands[k]‖² <= ball_radii[k]², so Lagrangian duality
# (Slater's condition is certified before this solver is called) states the
# worst-case newsvendor problem over the intersection, on the normalized
# support [0,1]^number_of_items, as
#
#   min_{λ ∈ R^K_+, order} Σ_k λ_k ball_radii[k]² + Σ_i sup_{ξ ∈ [0,1]}
#     [max(uᵢ (ξ - orderᵢ), oᵢ (orderᵢ - ξ)) - Σ_k λ_k (ξ - demands[k][i])²],
#
# where the supremum decomposes across items because both the loss and the
# squared transport cost separate coordinate-wise. The evaluation below rests
# on four observations, stated with Λ = Σ_k λ_k and the barycenter
# cᵢ = Σ_k λ_k demands[k][i] / Λ, which lies in [0,1].
#
# 1. Completing the square,
#
#      Σ_k λ_k (ξ - demands[k][i])² =
#          Λ (ξ - cᵢ)² + Σ_k λ_k demands[k][i]² - Λ cᵢ²,
#
#    so each supremum is the single-ball supremum at multiplier Λ and demand
#    cᵢ, minus the λ-dependent constant Σ_k λ_k demands[k][i]² - Λ cᵢ².
#
# 2. Per loss piece, sup_{ξ ∈ [0,1]} [slope ξ - Λ (ξ - cᵢ)²] is attained at
#    the unconstrained maximizer cᵢ + slope / (2Λ) pushed back into [0,1];
#    since cᵢ ∈ [0,1], only the bound in the slope's direction can bind, so
#    the atoms are cᵢ + min(1 - cᵢ, uᵢ / 2Λ) and cᵢ - min(cᵢ, oᵢ / 2Λ). At
#    Λ = 0 the caps are infinite (division by zero), the atoms sit at the box
#    corners 1 and 0, the quadratic terms vanish, and cᵢ is immaterial.
#
# 3. Writing Aᵢ and Bᵢ for the supremum values of the underage and overage
#    pieces, the order minimizing max(Aᵢ - uᵢ orderᵢ, Bᵢ + oᵢ orderᵢ) is the
#    crossing point (Aᵢ - Bᵢ) / (uᵢ + oᵢ), and no clamping to [0,1] is
#    needed: evaluating the suprema at ξ = cᵢ and bounding the quadratic by
#    zero gives uᵢ cᵢ <= Aᵢ <= uᵢ and -oᵢ <= -oᵢ cᵢ <= Bᵢ <= 0, hence
#    0 <= Aᵢ - Bᵢ <= uᵢ + oᵢ.
#
# 4. By Danskin's theorem, a subgradient of the dual objective in λ_k is
#    ball_radii[k]² - E_P ‖ξ - demands[k]‖² for any distribution P that
#    attains the suprema at the chosen order. Mixing the two atoms with
#    weights oᵢ / (uᵢ + oᵢ) and uᵢ / (uᵢ + oᵢ) puts the crossing point at
#    the critical fractile of that two-point marginal, so this P is such a
#    worst case (on the boundary cases orderᵢ ∈ {0, 1} it is one selection
#    from the subdifferential interval). The expectation uses the squared
#    distances to the original centers demands[k], not the recentered form,
#    so the λ-derivative of the constant from observation 1 is included.


# Evaluates the dual objective at λ, writing the minimizing normalized order
# into order and the subgradient into gradient. Numbered comments refer to
# the observations above.
function _intersection_dual_objective_and_gradient!(
    gradient,
    order,
    λ,
    normalized_demands,
    normalized_ball_radii,
    instance_underage_costs,
    instance_overage_costs,
)
    K = length(normalized_demands)
    total_multiplier = sum(λ)
    objective = 0.0
    for k in 1:K
        objective += λ[k] * normalized_ball_radii[k]^2
        gradient[k] = normalized_ball_radii[k]^2
    end

    for i in 1:number_of_items
        underage_cost = instance_underage_costs[i]
        overage_cost = instance_overage_costs[i]
        total_cost = underage_cost + overage_cost

        weighted_demand = 0.0
        weighted_squared_demand = 0.0
        for k in 1:K
            weighted_demand += λ[k] * normalized_demands[k][i]
            weighted_squared_demand += λ[k] * normalized_demands[k][i]^2
        end
        center = total_multiplier > 0.0 ?
            weighted_demand / total_multiplier : 0.5

        # Observation 2: the supremum of each loss piece is at its atom.
        upper_atom = center +
            min(1.0 - center, underage_cost / (2.0 * total_multiplier))
        lower_atom = center -
            min(center, overage_cost / (2.0 * total_multiplier))
        underage_value = underage_cost * upper_atom -
            total_multiplier * (upper_atom - center)^2
        overage_value = -overage_cost * lower_atom -
            total_multiplier * (lower_atom - center)^2

        # Observation 3: the optimal order is the crossing of the two pieces;
        # the trailing terms are the constant from observation 1.
        order[i] = (underage_value - overage_value) / total_cost
        crossing_value = (
            overage_cost * underage_value +
            underage_cost * overage_value
        ) / total_cost
        objective += crossing_value +
            total_multiplier * center^2 - weighted_squared_demand

        # Observation 4: the worst case mixes the atoms at the fractile.
        upper_weight = overage_cost / total_cost
        lower_weight = underage_cost / total_cost
        for k in 1:K
            gradient[k] -=
                upper_weight * (upper_atom - normalized_demands[k][i])^2 +
                lower_weight * (lower_atom - normalized_demands[k][i])^2
        end
    end
    return objective
end


# Minimizes the convex dual objective over λ ≥ 0. Unlike the single-ball
# dual, whose scalar multiplier admits a golden-section search, the
# intersection dual has one multiplier per ball, coupled through the
# barycenter, so no bracketed one-dimensional search applies. Instead:
# projected gradient descent with Barzilai-Borwein steps and Armijo
# backtracking, the monotone variant of the spectral projected gradient
# method ("Nonmonotone Spectral Projected Gradient Methods on Convex Sets"
# by Birgin, Martínez, and Raydan). The Barzilai-Borwein step
# ‖Δλ‖² / (Δλ ⋅ Δgradient) estimates the inverse curvature along the last
# step, which keeps the iteration count low even when a small intersection
# makes the optimal multipliers large. The line search is sound because the
# objective is continuously differentiable for Λ > 0: where a displacement
# cap in observation 2 switches branch, the two branches agree in value and
# first derivative. The iteration starts at λ = 0 and stops once the
# projected step is no longer a descent direction, which happens only at
# (numerical) stationarity; with large radii it stops immediately, recovering
# the distribution-free newsvendor.
function _solve_intersection_dual(
    normalized_demands,
    normalized_ball_radii,
    instance_underage_costs,
    instance_overage_costs,
)
    K = length(normalized_demands)
    λ = zeros(K)
    trial_λ = similar(λ)
    gradient = similar(λ)
    trial_gradient = similar(λ)
    direction = similar(λ)
    order = zeros(number_of_items)

    evaluate!(gradient_buffer, multipliers) =
        _intersection_dual_objective_and_gradient!(
            gradient_buffer,
            order,
            multipliers,
            normalized_demands,
            normalized_ball_radii,
            instance_underage_costs,
            instance_overage_costs,
        )

    objective = evaluate!(gradient, λ)
    step_scale = 1.0 / max(1.0e-12, sqrt(sum(abs2, gradient)))
    for _ in 1:2000
        directional_derivative = 0.0
        for k in 1:K
            direction[k] =
                max(0.0, λ[k] - step_scale * gradient[k]) - λ[k]
            directional_derivative += gradient[k] * direction[k]
        end
        directional_derivative < -1.0e-6 * (1.0 + abs(objective)) || break

        step = 1.0
        accepted = false
        trial_objective = Inf
        for _ in 1:40
            for k in 1:K
                trial_λ[k] = λ[k] + step * direction[k]
            end
            trial_objective = evaluate!(trial_gradient, trial_λ)
            if trial_objective <=
               objective + 1.0e-4 * step * directional_derivative
                accepted = true
                break
            end
            step /= 2.0
        end
        accepted || break

        step_inner_product = 0.0
        squared_step_norm = 0.0
        for k in 1:K
            step_difference = trial_λ[k] - λ[k]
            step_inner_product +=
                step_difference * (trial_gradient[k] - gradient[k])
            squared_step_norm += step_difference^2
            λ[k] = trial_λ[k]
            gradient[k] = trial_gradient[k]
        end
        objective = trial_objective
        step_scale = step_inner_product > 0.0 ?
            clamp(squared_step_norm / step_inner_product, 1.0e-12, 1.0e12) :
            min(step_scale * 10.0, 1.0e12)
    end

    # Rewrite order for the final λ; rejected trials may have overwritten it.
    objective = evaluate!(gradient, λ)
    return objective, order
end


function _normalized_intersection_objective_value_and_order(
    normalized_demands,
    normalized_ball_radii,
    pair_distances,
    instance_underage_costs,
    instance_overage_costs,
)
    active_indices = _nonredundant_ball_indices(
        normalized_ball_radii, pair_distances,
    )
    active_demands = view(normalized_demands, active_indices)
    active_radii = view(normalized_ball_radii, active_indices)
    active_distances = view(pair_distances, active_indices, active_indices)

    # An empty or touching intersection is repaired exactly as in the conic
    # formulation: every ball grows by the smallest shared radius increase that
    # makes the intersection nonempty, collapsing the ambiguity set to the
    # contact point. This is settled before the dual solver runs — the dual
    # is unbounded below when the intersection is empty — and a strictly
    # interior certificate is Slater's condition for the dual.
    geometry, critical_indices = _certified_two_ball_radius_increase(
        active_demands, active_radii, active_distances,
    )
    isnothing(geometry) && (geometry = _active_set_radius_increase(
        active_demands, active_radii, critical_indices,
    ))
    minimum_increase, point = geometry

    # At first contact the repaired ambiguity set is the point mass at point;
    # ordering exactly that demand incurs zero loss.
    if minimum_increase >= -multi_item_geometry_tolerance
        return 0.0, point
    end

    return _solve_intersection_dual(
        active_demands,
        active_radii,
        instance_underage_costs,
        instance_overage_costs,
    )
end


function REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
    ε,
    demands,
    weights,
    instance_underage_costs,
    instance_overage_costs,
)
    K = length(demands)
    normalized_demands = [demand ./ number_of_consumers for demand in demands]
    normalized_ball_radii =
        REMK_intersection_ball_radii(K, ε, weights[end]) ./
        number_of_consumers
    pair_distances = _pairwise_distances(normalized_demands)
    objective, order = _normalized_intersection_objective_value_and_order(
        normalized_demands,
        normalized_ball_radii,
        pair_distances,
        instance_underage_costs,
        instance_overage_costs,
    )
    return number_of_consumers * objective, number_of_consumers .* order
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
    # Centers do not change across this grid, so compute their distances once;
    # every radius and weight combination uses the same exact values.
    pair_distances = _pairwise_distances(normalized_demands)
    result_type = Tuple{Float64,Vector{Float64}}
    results = Matrix{result_type}(
        undef, length(ambiguity_radii), length(weight_vectors),
    )

    for weight_index in eachindex(weight_vectors)
        weights = weight_vectors[weight_index]
        for radius_index in eachindex(ambiguity_radii)
            ε = ambiguity_radii[radius_index]
            normalized_ball_radii = REMK_intersection_ball_radii(
                K, ε, weights[end],
            ) ./ number_of_consumers
            objective, order =
                _normalized_intersection_objective_value_and_order(
                normalized_demands,
                normalized_ball_radii,
                pair_distances,
                instance_underage_costs,
                instance_overage_costs,
            )
            results[radius_index, weight_index] =
                number_of_consumers * objective,
                number_of_consumers .* order
        end
    end
    return results
end
