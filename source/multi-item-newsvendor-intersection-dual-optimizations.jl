# Fast solver for the intersection approach of multi-item-newsvendor-intersection-conic-optimizations.jl.
# The ambiguity set is the intersection of the 2-Wasserstein balls of radii REMK_intersection_ball_radii
# around the K demands. If the balls do not intersect then their radii are all increased by the
# smallest amount that makes them intersect.


using LinearAlgebra


# Balls that overlap by less than this are treated as touching.
const multi_item_geometry_tolerance = 1.0e-6


# Same as norm(first_point - second_point) without allocating the difference first_point - second_point.
function _euclidean_distance(first_point, second_point)

        squared_distance = 0.0
        @inbounds for i in eachindex(first_point, second_point); squared_distance += (first_point[i] - second_point[i])^2; end

        return sqrt(squared_distance)
end


function _pairwise_distances(demands)

        K = length(demands)
        distances = zeros(K, K)
        for j in 1:K, k in j+1:K; distances[j, k] = distances[k, j] = _euclidean_distance(demands[j], demands[k]); end

        return distances
end


# Drops every ball that contains another ball as it does not constrain the intersection. This does
# not change when the same amount is added to every radius.
function _nonredundant_ball_indices(ball_radii, pair_distances)

        kept_indices = Int[]
        # Start from the smallest ball as the radii are nonincreasing.
        for j in Iterators.reverse(eachindex(ball_radii))
                contained_ball_already_kept = any(pair_distances[j, k] + ball_radii[k] <= ball_radii[j] for k in kept_indices)
                contained_ball_already_kept || push!(kept_indices, j)

        end

        return sort!(kept_indices)
end


# Smallest increase in the radii that puts the point in every ball (negative if it is already
# strictly inside every ball).
function _required_radius_increase_at_point(point, demands, ball_radii)

        return maximum(_euclidean_distance(point, demands[k]) - ball_radii[k] for k in eachindex(demands))
end


# Tries to find the smallest increase δ in the radii from two balls alone.
#
# Balls j and k only intersect for δ >= (‖ d_j - d_k ‖ - r_j - r_k)/2 and no radius can be negative
# so δ >= -min(r). Takes the pair of balls (or the smallest ball) with the largest bound and the
# point between their centres where they first touch. If this point is strictly inside every ball
# then the balls overlap and nothing more is needed. If it is in every ball once the radii are
# increased by the bound then the bound is the smallest increase. Otherwise more than two balls
# matter and nothing is returned.
#
# Also returns the pair for the active-set method to start from.
function _certified_two_ball_radius_increase(demands, ball_radii, pair_distances)

        K = length(demands)
        lower_bound = -minimum(ball_radii)
        critical_j = 0
        critical_k = argmin(ball_radii)
        for j in 1:K, k in j+1:K
                bound = (pair_distances[j, k] - ball_radii[j] - ball_radii[k]) / 2.0
                if bound > lower_bound
                        lower_bound = bound
                        critical_j, critical_k = j, k
                end

        end
        critical_indices = critical_j == 0 ? [critical_k] : [critical_j, critical_k]

        candidate = if critical_j == 0
                demands[critical_k]
        else
                distance = pair_distances[critical_j, critical_k]
                distance > 0.0 || return nothing, critical_indices
                fraction = (ball_radii[critical_j] + lower_bound) / distance
                0.0 <= fraction <= 1.0 || return nothing, critical_indices
                demands[critical_j] + fraction * (demands[critical_k] - demands[critical_j])
        end
        candidate_increase = _required_radius_increase_at_point(candidate, demands, ball_radii)
        tolerance = 1.0e-6 * max(1.0, abs(lower_bound), abs(candidate_increase))
        # A point strictly inside every ball is enough so its increase is returned even if it is not the
        # smallest. Otherwise the bound is the smallest increase only if the point is in every ball once the
        # radii grow by it.
        candidate_increase < -tolerance && return (candidate_increase, candidate), critical_indices
        candidate_increase <= lower_bound + tolerance || return nothing, critical_indices

        return (max(lower_bound, candidate_increase), candidate), critical_indices
end


# Smallest δ for which a point ξ has ‖ ξ - d_k ‖ = r_k + δ for every ball k of the support. Returns
# δ and ξ.
#
# ξ lies in the affine hull of the centres. Subtracting the square of the first equation from the
# squares of the others leaves equations that are linear in ξ and δ. The first equation is then a
# quadratic in δ.
function _ball_from_support(demands, ball_radii, support)

        first_index = first(support)
        first_demand = demands[first_index]
        first_radius = ball_radii[first_index]
        length(support) == 1 && return -first_radius, copy(first_demand)

        other_indices = support[2:end]
        differences = Matrix{Float64}(undef, length(first_demand), length(other_indices))
        for (column, index) in enumerate(other_indices); differences[:, column] = demands[index] - first_demand; end
        gram = differences' * differences
        radius_differences = ball_radii[other_indices] .- first_radius
        right_hand_sides = hcat(0.5 .* (diag(gram) .- (ball_radii[other_indices] .^ 2 .- first_radius^2)), -radius_differences)
        # Solve the linear equations for the point in terms of δ. Returns nothing if the centres are
        # affinely dependent.
        factorization = cholesky(Symmetric(gram); check = false)
        issuccess(factorization) || return nothing
        coordinates = factorization \ right_hand_sides
        constant_coordinates = coordinates[:, 1]
        increase_coordinates = coordinates[:, 2]
        constant_offset = differences * constant_coordinates
        increase_offset = differences * increase_coordinates

        quadratic = dot(increase_offset, increase_offset) - 1.0
        linear = 2.0 * (dot(constant_offset, increase_offset) - first_radius)
        constant = dot(constant_offset, constant_offset) - first_radius^2
        roots = if abs(quadratic) <= 1.0e-6
                abs(linear) <= 1.0e-6 && return nothing
                (-constant / linear,)
        else
                discriminant = linear^2 - 4.0 * quadratic * constant
                discriminant >= -1.0e-6 || return nothing
                root_offset = sqrt(max(0.0, discriminant))
                ((-linear - root_offset) / (2.0 * quadratic), (-linear + root_offset) / (2.0 * quadratic))
        end

        valid_roots = [root for root in roots if all(root + ball_radii[index] >= -1.0e-6 for index in support)]
        isempty(valid_roots) && return nothing
        increase = minimum(valid_roots)
        point = first_demand + constant_offset + increase * increase_offset

        return increase, point
end


# Smallest increase δ for which the balls of the support share a point. Returns δ and the point. The
# point touches some of the balls and is inside the others so every subset is tried as the balls it
# touches (the support is only ever a few balls).
function _smallest_supported_ball(demands, ball_radii, support)

        isempty(support) && return -Inf, zeros(length(first(demands)))
        best_geometry = (Inf, zeros(length(first(demands))))
        for mask in 1:(1 << length(support)) - 1
                subset = [support[j] for j in eachindex(support) if mask & (1 << (j - 1)) != 0]
                geometry = _ball_from_support(demands, ball_radii, subset)
                isnothing(geometry) && continue
                increase, point = geometry
                increase < best_geometry[1] || continue
                all(_euclidean_distance(point, demands[index]) <= increase + ball_radii[index] + 1.0e-6 for index in support) && (best_geometry = geometry)

        end

        return best_geometry
end


# Smallest increase for the first count balls of indices together with the balls of the support.
# This follows Welzl's algorithm for the smallest enclosing ball (doi:10.1007/BFb0038202; see also
# Fischer and Gärtner doi:10.1142/S0218195904001500).
function _active_set_radius_increase(demands, ball_radii, indices, support, count)

        # Stops when no balls are left or once the support has number_of_items + 1 balls without looking at
        # the balls left.
        if count == 0 || length(support) == length(first(demands)) + 1; return _smallest_supported_ball(demands, ball_radii, support); end

        # Solve without the last ball and keep the result if the point is in that ball.
        geometry = _active_set_radius_increase(demands, ball_radii, indices, support, count - 1)
        increase, point = geometry
        index = indices[count]
        _euclidean_distance(point, demands[index]) <= increase + ball_radii[index] + 1.0e-6 && return geometry

        # Otherwise solve again with the last ball in the support.
        push!(support, index)
        geometry = _active_set_radius_increase(demands, ball_radii, indices, support, count - 1)
        pop!(support)

        return geometry
end


# Smallest increase found by solving for a few balls at a time. Each step adds the ball furthest
# outside the point and keeps the balls the point touches until the point is in every ball. Only
# used when the recursion above fails.
function _pivoting_radius_increase(demands, ball_radii)

        support = [argmin(ball_radii)]
        for _ in 1:1000
                increase, point = _smallest_supported_ball(demands, ball_radii, support)
                isfinite(increase) || break
                excesses = [_euclidean_distance(point, demands[k]) - ball_radii[k] - increase for k in eachindex(demands)]
                furthest = argmax(excesses)
                excesses[furthest] <= 1.0e-6 && return increase, point
                support = [k for k in support if excesses[k] >= -1.0e-6]
                push!(support, furthest)

        end
        error("The smallest radius increase was not found.")
end


# Smallest increase by the recursion above with the preferred balls first to enter the support.
function _active_set_radius_increase(demands, ball_radii, preferred_indices = Int[])

        indices = collect(eachindex(demands))
        indices = vcat(preferred_indices, filter(index -> index ∉ preferred_indices, indices))
        increase, point = _active_set_radius_increase(demands, ball_radii, indices, Int[], length(demands))

        # Check that the point is in every ball and otherwise solve again by pivoting.
        if any(_euclidean_distance(point, demands[k]) > increase + ball_radii[k] + 1.0e-6 for k in eachindex(demands)); return _pivoting_radius_increase(demands, ball_radii); end

        return increase, point
end


# Dual problem for overlapping balls. Theorem 2 of Rychener et al. gives (for demands scaled to [0,1])
#
#       min Σ_k λ[k]*radii[k]^2 + Σ_i sup_ξ {max(u[i]*(ξ - order[i]), o[i]*(order[i] - ξ)) - Σ_k λ[k]*(ξ - d[k][i])^2}
#
# over λ ≥ 0 and the order. Here u and o are the underage and overage costs and d[k] is the k-th
# demand. Each supremum is over ξ in [0,1]. Completing the square gives
#
#       Σ_k λ[k]*(ξ - d[k][i])^2 = Λ*(ξ - c)^2 + Σ_k λ[k]*d[k][i]^2 - Λ*c^2
#
# for Λ = Σ_k λ[k] and c = Σ_k λ[k]*d[k][i]/Λ. So each supremum is that of a single ball with
# multiplier Λ and demand c.
#
# Underage and overage pieces attain their suprema at the upper atom c + min(1 - c, u[i]/(2Λ)) and
# the lower atom c - min(c, o[i]/(2Λ)). The best order is where the two pieces cross. The gradient
# in λ[k] is radii[k]^2 minus the expected squared distance to d[k] under the worst-case
# distribution. For item i this puts probability o[i]/(u[i] + o[i]) on the upper atom and
# u[i]/(u[i] + o[i]) on the lower atom.


# Dual objective value at λ.
function _intersection_dual_objective_and_gradient!(gradient, order, λ, normalized_demands, normalized_ball_radii, instance_underage_costs, instance_overage_costs)

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
                center = total_multiplier > 0.0 ? weighted_demand / total_multiplier : 0.5

                # Suprema of the underage and overage pieces.
                upper_atom = center + min(1.0 - center, underage_cost / (2.0 * total_multiplier))
                lower_atom = center - min(center, overage_cost / (2.0 * total_multiplier))
                underage_value = underage_cost * upper_atom - total_multiplier * (upper_atom - center)^2
                overage_value = -overage_cost * lower_atom - total_multiplier * (lower_atom - center)^2

                # Order where the two pieces cross and their value there plus the terms left from completing the square.
                order[i] = (underage_value - overage_value) / total_cost
                crossing_value = (overage_cost * underage_value + underage_cost * overage_value) / total_cost
                objective += crossing_value + total_multiplier * center^2 - weighted_squared_demand

                # Worst-case distribution of the item on the two atoms and its part of the gradient.
                upper_weight = overage_cost / total_cost
                lower_weight = underage_cost / total_cost
                for k in 1:K; gradient[k] -= upper_weight * (upper_atom - normalized_demands[k][i])^2 + lower_weight * (lower_atom - normalized_demands[k][i])^2; end

        end

        return objective
end


# Spectral projected gradient method of Birgin et al. (doi:10.1137/S1052623497330963; see also
# Barzilai and Borwein doi:10.1093/imanum/8.1.141) minimizing the dual objective over λ ≥ 0 with a
# monotone line search. Starts from λ = 0 which is already optimal for large radii.
function _solve_intersection_dual(normalized_demands, normalized_ball_radii, instance_underage_costs, instance_overage_costs)

        K = length(normalized_demands)
        λ = zeros(K)
        trial_λ = similar(λ)
        gradient = similar(λ)
        trial_gradient = similar(λ)
        direction = similar(λ)
        order = zeros(number_of_items)

        evaluate!(gradient_buffer, multipliers) = _intersection_dual_objective_and_gradient!(gradient_buffer, order, multipliers, normalized_demands, normalized_ball_radii, instance_underage_costs, instance_overage_costs)

        objective = evaluate!(gradient, λ)
        step_scale = 1.0 / max(1.0e-12, sqrt(sum(abs2, gradient)))
        for _ in 1:2000

                directional_derivative = 0.0
                for k in 1:K
                        direction[k] = max(0.0, λ[k] - step_scale * gradient[k]) - λ[k]
                        directional_derivative += gradient[k] * direction[k]

                end

                # Stop once the predicted decrease and the projected gradient are both small. The objective is flat
                # near the minimum but the order is not. With these tolerances the orders are within about 0.03% of
                # those for far tighter tolerances.
                if directional_derivative >= -1.0e-8 * (1.0 + abs(objective))
                        # Projected gradient for a unit step. Unlike the predicted decrease it does not depend on step_scale.
                        projected_gradient_norm = sqrt(sum((λ[k] - max(0.0, λ[k] - gradient[k]))^2 for k in 1:K))
                        projected_gradient_norm <= 1.0e-7 && break
                end

                step = 1.0
                accepted = false
                trial_objective = Inf
                for _ in 1:40
                        for k in 1:K; trial_λ[k] = λ[k] + step * direction[k]; end
                        trial_objective = evaluate!(trial_gradient, trial_λ)
                        if trial_objective <= objective + 1.0e-4 * step * directional_derivative; accepted = true; break; end
                        step /= 2.0

                end
                accepted || break

                step_inner_product = 0.0
                squared_step_norm = 0.0
                for k in 1:K
                        step_difference = trial_λ[k] - λ[k]
                        step_inner_product += step_difference * (trial_gradient[k] - gradient[k])
                        squared_step_norm += step_difference^2
                        λ[k] = trial_λ[k]
                        gradient[k] = trial_gradient[k]

                end
                objective = trial_objective
                step_scale = step_inner_product > 0.0 ? clamp(squared_step_norm / step_inner_product, 1.0e-12, 1.0e12) : min(step_scale * 10.0, 1.0e12)

        end

        # Rewrite the order for the final λ as it was last written for a trial step.
        objective = evaluate!(gradient, λ)

        return objective, order
end


# Intersection approach for demands and radii already scaled for numerical stability.
function _normalized_intersection_objective_value_and_order(normalized_demands, normalized_ball_radii, pair_distances, instance_underage_costs, instance_overage_costs)

        # Drop the balls that contain another ball.
        active_indices = _nonredundant_ball_indices(normalized_ball_radii, pair_distances)
        active_demands = view(normalized_demands, active_indices)
        active_radii = view(normalized_ball_radii, active_indices)
        active_distances = view(pair_distances, active_indices, active_indices)

        # Smallest increase in the radii for which the balls share a point. If they overlap the point may
        # instead be any point strictly inside every ball. Euclidean balls give the same answer as
        # Wasserstein balls because a distribution within r of the point mass at d has
        # ‖ mean - d ‖^2 + E‖ demand - mean ‖^2 <= r^2.
        geometry, critical_indices = _certified_two_ball_radius_increase(active_demands, active_radii, active_distances)
        isnothing(geometry) && (geometry = _active_set_radius_increase(active_demands, active_radii, critical_indices))
        minimum_increase, point = geometry

        # If the balls only meet after the increase then the only distribution left is the point mass at the
        # shared point. In this case, ordering this point costs nothing as the demand is this point for certain.
        if minimum_increase >= -multi_item_geometry_tolerance; return 0.0, point; end

        return _solve_intersection_dual(active_demands, active_radii, instance_underage_costs, instance_overage_costs)
end


# 2-Wasserstein DRO over the intersection of the balls around the demands.
function REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(ε, demands, weights, instance_underage_costs, instance_overage_costs)

        # Demands and radii divided by the number of consumers to keep the problem numerically stable
        # (undone on return).
        K = length(demands)
        normalized_demands = [demand ./ number_of_consumers for demand in demands]
        normalized_ball_radii = REMK_intersection_ball_radii(K, ε, weights[end]) ./ number_of_consumers # Every entry of weights is ρ╱ε (see REMK_intersection_weights).

        pair_distances = _pairwise_distances(normalized_demands)
        objective, order = _normalized_intersection_objective_value_and_order(normalized_demands, normalized_ball_radii, pair_distances, instance_underage_costs, instance_overage_costs)

        return number_of_consumers * objective, number_of_consumers .* order
end


# Intersection version of _multi_item_newsvendor_grid that computes the demand distances only once.
function _multi_item_newsvendor_grid(::typeof(REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order), ambiguity_radii, demands, weight_vectors, instance_underage_costs, instance_overage_costs)

        K = length(demands)
        normalized_demands = [demand ./ number_of_consumers for demand in demands]
        pair_distances = _pairwise_distances(normalized_demands)
        result_type = Tuple{Float64,Vector{Float64}}
        results = Matrix{result_type}(undef, length(ambiguity_radii), length(weight_vectors))

        for weight_index in eachindex(weight_vectors)
                weights = weight_vectors[weight_index]
                for radius_index in eachindex(ambiguity_radii)
                        ε = ambiguity_radii[radius_index]
                        normalized_ball_radii = REMK_intersection_ball_radii(K, ε, weights[end]) ./ number_of_consumers # Every entry of weights is ρ╱ε (see REMK_intersection_weights).
                        objective, order = _normalized_intersection_objective_value_and_order(normalized_demands, normalized_ball_radii, pair_distances, instance_underage_costs, instance_overage_costs)
                        results[radius_index, weight_index] = number_of_consumers * objective, number_of_consumers .* order

                end
        end

        return results
end
