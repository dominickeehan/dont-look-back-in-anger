# If the Wasserstein balls do not intersect, their radii are increased by the
# smallest common additive amount that makes the intersection nonempty.
#
# REMK_intersection_weights repeats ρ/ε to match the common solver interface,
# so the final entry supplies the radius ratio.


# Conic formulation for an intersection of Wasserstein balls.


# Since W₂(P, δ_d)² = ‖E_P[ξ] - d‖² + tr(Cov(P)), the point mass at
# E_P[ξ] strongly dominates P whenever its covariance is nonzero. Feasibility
# therefore reduces to intersecting Euclidean balls. When an empty intersection
# is enlarged to first contact, the Euclidean intersection is a singleton and
# the ambiguity set contains only the point mass there.
function _build_ball_intersection_feasibility_problem(demands, ball_radii)
    K = length(demands)
    Problem = _new_multi_item_model()
    @variables(Problem, begin
        1.0 >= ξ[i = 1:number_of_items] >= 0.0
        a
    end)

    ball_constraints = ConstraintRef[]
    for k in 1:K
        push!(
            ball_constraints,
            @constraint(
                Problem,
                [
                    ball_radii[k] + a;
                    [
                        ξ[i] - demands[k][i]
                        for i in 1:number_of_items
                    ]
                ] in MathOptInterface.SecondOrderCone(
                    number_of_items + 1,
                ),
            ),
        )
    end

    @objective(Problem, Min, a)
    return Problem, ξ, a, ball_constraints
end


function _solve_ball_intersection_feasibility_problem!(
    Problem, ξ, a,
)
    _optimize_multi_item_model!(Problem)
    return value(a), value.(ξ)
end


# Specialize Corollary 2 to K point-mass reference distributions and
# the multi-item newsvendor loss.
function _build_intersection_W2_DRO_multi_item_newsvendor_problem(
    demands,
    ball_radii,
    instance_underage_costs,
    instance_overage_costs,
)
    a, b, C, g = _multi_item_newsvendor_problem_data(
        instance_underage_costs, instance_overage_costs,
    )
    K = length(demands)

    Problem = _new_multi_item_model()
    @variables(Problem, begin
        1.0 >= order[i = 1:number_of_items] >= 0.0
        λ[k = 1:K] >= 0.0
        eta
        z[l = 1:length(a), m = 1:length(g)] >= 0.0
        w[l = 1:length(a), k = 1:K, i = 1:number_of_items]
        s[l = 1:length(a), k = 1:K] >= 0.0
    end)

    # The rotated cone represents the quadratic-over-linear term
    # ‖w‖² / (4λ).
    for l in eachindex(a)
        @constraint(
            Problem,
            sum(b[l][i] * order[i] for i in 1:number_of_items) +
            sum(
                w[l, k, i] * demands[k][i]
                for k in 1:K, i in 1:number_of_items
            ) +
            sum(s[l, k] for k in 1:K) +
            sum(z[l, m] * g[m] for m in eachindex(g)) <= eta,
        )

        for i in 1:number_of_items
            @constraint(
                Problem,
                a[l][i] -
                sum(C[m, i] * z[l, m] for m in eachindex(g)) ==
                sum(w[l, k, i] for k in 1:K),
            )
        end

        for k in 1:K
            @constraint(
                Problem,
                [
                    2.0 * λ[k];
                    s[l, k];
                    [w[l, k, i] for i in 1:number_of_items]
                ] in MathOptInterface.RotatedSecondOrderCone(
                    number_of_items + 2,
                ),
            )
        end
    end

    @objective(
        Problem,
        Min,
        sum(ball_radii[k]^2 * λ[k] for k in 1:K) + eta,
    )
    return Problem, order
end


function _solve_intersection_W2_DRO_multi_item_newsvendor_problem!(
    Problem, order,
)
    _optimize_multi_item_model!(Problem)
    return (
        number_of_consumers * objective_value(Problem),
        number_of_consumers .* value.(order),
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

    Ball_Intersection_Feasibility_Problem,
    ξ,
    a,
    _ = _build_ball_intersection_feasibility_problem(
        normalized_demands, normalized_ball_radii,
    )
    minimum_increase, point =
        _solve_ball_intersection_feasibility_problem!(
            Ball_Intersection_Feasibility_Problem, ξ, a,
        )

    # Treat intersections within the geometry tolerance as first contact. At
    # first contact, the only demand distribution in the ambiguity set is the
    # point mass at the contact point. Ordering this singleton demand gives
    # zero loss.
    if minimum_increase >= -multi_item_geometry_tolerance
        return 0.0, number_of_consumers .* point
    end

    Problem, order =
        _build_intersection_W2_DRO_multi_item_newsvendor_problem(
            normalized_demands,
            normalized_ball_radii,
            instance_underage_costs,
            instance_overage_costs,
        )
    return _solve_intersection_W2_DRO_multi_item_newsvendor_problem!(
        Problem, order,
    )
end
