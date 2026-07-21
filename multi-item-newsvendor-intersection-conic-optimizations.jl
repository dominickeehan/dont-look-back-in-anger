# Exact conic formulation for an intersection of Wasserstein balls.


# Construct the affine pieces of the newsvendor loss and the normalized box
# support [0,1]^number_of_items.
# See Corollary 2 of
# "Wasserstein Distributionally Robust Optimization with Heterogeneous Data
# Sources" by Rychener, Esteban-Perez, Morales, and Kuhn.
function _multi_item_newsvendor_problem_data(
    instance_underage_costs, instance_overage_costs,
)
    loss_pieces = collect(
        Iterators.product(fill([false, true], number_of_items)...),
    )
    a = [
        [
            loss_pieces[l][i] ? instance_underage_costs[i] :
            -instance_overage_costs[i]
            for i in 1:number_of_items
        ]
        for l in eachindex(loss_pieces)
    ]
    b = [-a[l] for l in eachindex(a)]
    C = [
        -Matrix{Float64}(I, number_of_items, number_of_items);
        Matrix{Float64}(I, number_of_items, number_of_items)
    ]
    g = [
        zeros(number_of_items);
        ones(number_of_items)
    ]
    return a, b, C, g
end


# In the following distributional ball-intersection feasibility problem the equivalence to working in R^m follows since
# W₂(P,1_ξ) = sqrt(sum((E(P)_i-ξ_i)^2) + Tr(Cov(P))) which drives Tr(Cov(P)) -> 0 and P -> 1_E(P) at extremal distributions.
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

    # At first contact the repaired ambiguity set is the point mass at point;
    # ordering exactly that demand incurs zero loss.
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
