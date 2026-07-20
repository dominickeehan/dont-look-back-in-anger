using Test
using LinearAlgebra
using Random


number_of_items = 3
number_of_consumers = 10.0
underage_costs = fill(4.0, number_of_items)
overage_costs = fill(1.0, number_of_items)
cu = underage_costs[1]
co = overage_costs[1]

include("weights.jl")
include("multi-item-newsvendor-optimizations.jl")


# Small exhaustive formulations retained as an exactness oracle. They should
# only be used for tests because their size is exponential in number_of_items.
reference_loss_pieces =
    collect(Iterators.product(fill([false, true], number_of_items)...))
reference_a = [
    [reference_loss_pieces[l][i] ? cu : -co for i in 1:number_of_items]
    for l in eachindex(reference_loss_pieces)
]
reference_b = [-reference_a[l] for l in eachindex(reference_a)]
reference_C = [
    -Matrix{Float64}(I, number_of_items, number_of_items);
    Matrix{Float64}(I, number_of_items, number_of_items)
]
reference_g =
    [zeros(number_of_items); number_of_consumers * ones(number_of_items)]


function reference_weighted_W2(epsilon, demands, weights)
    weights = weights ./ sum(weights)
    T = length(demands)
    L = length(reference_a)
    problem = _new_multi_item_model()

    @variables(problem, begin
        number_of_consumers >= order[i = 1:number_of_items] >= 0.0
        lambda >= 0.0
        gamma[t = 1:T]
        z[t = 1:T, l = 1:L, m = 1:length(reference_g)] >= 0.0
        w[t = 1:T, l = 1:L, i = 1:number_of_items]
    end)

    for t in 1:T, l in 1:L
        @constraint(
            problem,
            [
                2.0 * lambda;
                gamma[t] -
                sum(reference_b[l][i] * order[i] for i in 1:number_of_items) -
                sum(w[t, l, i] * demands[t][i] for i in 1:number_of_items) -
                sum(z[t, l, m] * reference_g[m] for m in eachindex(reference_g));
                [w[t, l, i] for i in 1:number_of_items]
            ] in MathOptInterface.RotatedSecondOrderCone(number_of_items + 2),
        )
        for i in 1:number_of_items
            @constraint(
                problem,
                reference_a[l][i] -
                sum(reference_C[m, i] * z[t, l, m] for m in eachindex(reference_g)) ==
                w[t, l, i],
            )
        end
    end

    @objective(
        problem,
        Min,
        epsilon^2 * lambda + sum(weights[t] * gamma[t] for t in 1:T),
    )
    _optimize_multi_item_model!(problem; high_precision = true)
    return objective_value(problem)
end


function reference_intersection_W2(epsilon, demands, radius_ratio)
    K = length(demands)
    L = length(reference_a)
    radii = REMK_intersection_ball_radii(K, epsilon, radius_ratio)
    problem = _new_multi_item_model()

    @variables(problem, begin
        number_of_consumers >= order[i = 1:number_of_items] >= 0.0
        lambda[k = 1:K] >= 0.0
        eta
        z[l = 1:L, m = 1:length(reference_g)] >= 0.0
        w[l = 1:L, k = 1:K, i = 1:number_of_items]
        s[l = 1:L, k = 1:K] >= 0.0
    end)

    for l in 1:L
        @constraint(
            problem,
            sum(reference_b[l][i] * order[i] for i in 1:number_of_items) +
            sum(
                w[l, k, i] * demands[k][i]
                for k in 1:K, i in 1:number_of_items
            ) +
            sum(s[l, k] for k in 1:K) +
            sum(z[l, m] * reference_g[m] for m in eachindex(reference_g)) <= eta,
        )
        for i in 1:number_of_items
            @constraint(
                problem,
                reference_a[l][i] -
                sum(reference_C[m, i] * z[l, m] for m in eachindex(reference_g)) ==
                sum(w[l, k, i] for k in 1:K),
            )
        end
        for k in 1:K
            @constraint(
                problem,
                [
                    2.0 * lambda[k];
                    s[l, k];
                    [w[l, k, i] for i in 1:number_of_items]
                ] in MathOptInterface.RotatedSecondOrderCone(number_of_items + 2),
            )
        end
    end

    @objective(
        problem,
        Min,
        sum(radii[k]^2 * lambda[k] for k in 1:K) + eta,
    )
    _optimize_multi_item_model!(problem)
    return objective_value(problem)
end


# Test-only oracle for the smallest epsilon with a nonempty intersection at a
# fixed radius profile; used to place randomized trial radii in the interior
# regime.
function reference_minimum_intersection_geometry(normalized_demands, relative_radii)
    problem = _new_multi_item_model()
    @variables(problem, begin
        1.0 >= feasible_point[i = 1:number_of_items] >= 0.0
        minimum_normalized_epsilon >= 0.0
    end)
    for k in eachindex(normalized_demands)
        @constraint(
            problem,
            [
                relative_radii[k] * minimum_normalized_epsilon;
                [
                    feasible_point[i] - normalized_demands[k][i]
                    for i in 1:number_of_items
                ]
            ] in MathOptInterface.SecondOrderCone(number_of_items + 1),
        )
    end
    @objective(problem, Min, minimum_normalized_epsilon)
    _optimize_multi_item_model!(problem; high_precision = true)
    return value(minimum_normalized_epsilon), value.(feasible_point)
end


function reference_minimum_radius_increase_geometry(
    normalized_demands, normalized_base_radii,
)
    problem = _new_multi_item_model()
    @variables(problem, begin
        1.0 >= feasible_point[i = 1:number_of_items] >= 0.0
        minimum_normalized_increase
    end)
    for k in eachindex(normalized_demands)
        @constraint(
            problem,
            [
                normalized_base_radii[k] + minimum_normalized_increase;
                [
                    feasible_point[i] - normalized_demands[k][i]
                    for i in 1:number_of_items
                ]
            ] in MathOptInterface.SecondOrderCone(number_of_items + 1),
        )
    end
    @objective(problem, Min, minimum_normalized_increase)
    _optimize_multi_item_model!(problem; high_precision = true)
    return value(minimum_normalized_increase), value.(feasible_point)
end


normalized_intersection_radii(K, normalized_epsilon, radius_ratio) =
    [normalized_epsilon * (1.0 + (K - k + 1) * radius_ratio) for k in 1:K]


demands = [
    [2.0, 4.0, 3.0],
    [5.0, 3.0, 6.0],
    [3.0, 6.0, 4.0],
    [4.0, 5.0, 7.0],
]
weights = [0.1, 0.2, 0.3, 0.4]


@testset "shared radius increase geometry" begin
    # Equal radii: the shared increase closes the gap symmetrically, so first
    # contact is the midpoint of the two centers.
    pair_demands = [zeros(number_of_items), [1.0, 0.0, 0.0]]
    equal_radii = [0.1, 0.1]
    reference_increase, _ = reference_minimum_radius_increase_geometry(
        pair_demands, equal_radii,
    )
    pair_certificate, _ = _certified_radius_increase_and_lower_bound(
        pair_demands, equal_radii,
    )
    minimum_increase, feasible_point =
        _compute_minimum_radius_increase_and_point(pair_demands, equal_radii)

    @test minimum_increase ≈ 0.4 atol = 1.0e-12
    @test minimum_increase ≈ reference_increase atol = 1.0e-8 rtol = 1.0e-8
    @test feasible_point ≈ [0.5, 0.0, 0.0] atol = 1.0e-12
    @test !isnothing(pair_certificate)

    # Unequal radii shift first contact toward the smaller ball, and the two
    # enlarged radii still meet exactly at the contact point.
    unequal_radii = [0.3, 0.1]
    minimum_increase, feasible_point =
        _compute_minimum_radius_increase_and_point(pair_demands, unequal_radii)
    @test minimum_increase ≈ 0.3 atol = 1.0e-12
    @test feasible_point ≈ [0.6, 0.0, 0.0] atol = 1.0e-12

    # A strictly interior intersection gives a negative increase: the balls
    # could be shrunk by that much and still meet.
    interior_radii = [0.6, 0.6]
    minimum_increase, feasible_point =
        _compute_minimum_radius_increase_and_point(pair_demands, interior_radii)
    @test minimum_increase ≈ -0.1 atol = 1.0e-12
    @test feasible_point ≈ [0.5, 0.0, 0.0] atol = 1.0e-12

    # A single ball inside the support box can shrink to its own center.
    single_increase, single_point =
        _compute_minimum_radius_increase_and_point([[0.5, 0.4, 0.5]], [0.2])
    @test single_increase ≈ -0.2 atol = 1.0e-12
    @test single_point ≈ [0.5, 0.4, 0.5] atol = 1.0e-12

    # The support box binds before the balls do.
    support_demands = [[-0.3, 0.4, 0.5]]
    support_increase, support_point =
        _compute_minimum_radius_increase_and_point(support_demands, [0.1])
    @test support_increase ≈ 0.2 atol = 1.0e-12
    @test support_point ≈ [0.0, 0.4, 0.5] atol = 1.0e-12

    # Three active balls have pairwise lower bounds below the true optimum, so
    # the compact high-precision SOCP must close the remaining geometry gap.
    target_point = fill(0.5, number_of_items)
    directions = [
        [1.0, 0.0, 0.0],
        [-0.5, sqrt(3.0) / 2.0, 0.0],
        [-0.5, -sqrt(3.0) / 2.0, 0.0],
    ]
    target_increase = 0.2
    target_radii = [0.7, 0.5, 0.3]
    three_active_demands = [
        target_point .+ (target_radii[k] + target_increase) .* directions[k]
        for k in 1:3
    ]
    reference_increase, _ = reference_minimum_radius_increase_geometry(
        three_active_demands, target_radii,
    )
    three_ball_certificate, _ = _certified_radius_increase_and_lower_bound(
        three_active_demands, target_radii,
    )
    minimum_increase, feasible_point =
        _compute_minimum_radius_increase_and_point(
            three_active_demands, target_radii,
        )

    @test minimum_increase ≈ target_increase atol = 1.0e-8 rtol = 1.0e-8
    @test minimum_increase ≈ reference_increase atol = 1.0e-8 rtol = 1.0e-8
    @test feasible_point ≈ target_point atol = 1.0e-7 rtol = 1.0e-7
    @test isnothing(three_ball_certificate)

    rng = MersenneTwister(314159)
    for _ in 1:10
        K = rand(rng, 1:6)
        trial_demands = [rand(rng, number_of_items) for _ in 1:K]
        trial_radii = normalized_intersection_radii(
            K, rand(rng) * 0.2, rand(rng) * 0.5,
        )
        reference_increase, _ = reference_minimum_radius_increase_geometry(
            trial_demands, trial_radii,
        )
        trial_increase, trial_point =
            _compute_minimum_radius_increase_and_point(trial_demands, trial_radii)
        @test trial_increase ≈ reference_increase atol = 1.0e-8 rtol = 1.0e-8
        @test all(value -> 0.0 <= value <= 1.0, trial_point)
        # The returned point must be feasible for the enlarged balls, and at
        # least one enlarged ball must be tight.
        slacks = [
            trial_radii[k] + trial_increase -
            sqrt(_squared_euclidean_distance(trial_point, trial_demands[k]))
            for k in 1:K
        ]
        @test minimum(slacks) >= -1.0e-8
        @test minimum(abs.(slacks)) <= 1.0e-7
    end
end


@testset "compact multi-item newsvendor formulations" begin
    weighted_reference_objective = reference_weighted_W2(1.2, demands, weights)
    compact_objective, compact_order =
        W2_DRO_multi_item_newsvendor_objective_value_and_order(
            1.2, demands, weights,
            underage_costs,
            overage_costs,
        )
    @test compact_objective ≈ weighted_reference_objective atol = 1.0e-3
    @test all((0.0 .<= compact_order) .& (compact_order .<= number_of_consumers))

    support_diameter = number_of_consumers * sqrt(number_of_items)
    saturated_objective, saturated_order =
        W2_DRO_multi_item_newsvendor_objective_value_and_order(
            support_diameter, demands, weights,
            underage_costs,
            overage_costs,
        )
    expected_saturated_objective =
        number_of_items * number_of_consumers * cu * co / (cu + co)
    @test saturated_objective ≈ expected_saturated_objective atol = 1.0e-10
    @test saturated_order ≈ fill(
        number_of_consumers * cu / (cu + co), number_of_items,
    )

    endpoint_demands = [zeros(number_of_items), fill(number_of_consumers, number_of_items)]
    endpoint_objective, endpoint_order =
        W2_DRO_multi_item_newsvendor_objective_value_and_order(
            1.0, endpoint_demands, [0.8, 0.2],
            underage_costs,
            overage_costs,
        )
    @test endpoint_objective ≈ expected_saturated_objective atol = 1.0e-10
    @test endpoint_order ≈ saturated_order atol = 1.0e-10

    tied_demands = [
        [2.0, 5.0, 0.0],
        [2.0, 5.0, 10.0],
        [8.0, 5.0, 0.0],
        [8.0, 5.0, 10.0],
    ]
    tied_weights = [0.4, 0.1, 0.1, 0.4]
    tied_reference_objective =
        reference_weighted_W2(1.5, tied_demands, tied_weights)
    tied_objective, _ =
        W2_DRO_multi_item_newsvendor_objective_value_and_order(
            1.5, tied_demands, tied_weights,
            underage_costs,
            overage_costs,
        )
    @test tied_objective ≈ tied_reference_objective atol = 3.0e-3

    radius_ratio = 0.2
    intersection_reference_objective =
        reference_intersection_W2(2.5, demands, radius_ratio)
    compact_objective, compact_order =
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
            2.5,
            demands,
            REMK_intersection_weights(length(demands), radius_ratio),
            underage_costs,
            overage_costs,
        )
    @test compact_objective ≈ intersection_reference_objective atol = 1.0e-3
    @test all((0.0 .<= compact_order) .& (compact_order .<= number_of_consumers))

    grid_radii = [4.0, 2.5, 3.25]
    grid_weights = [
        REMK_intersection_weights(length(demands), 0.2),
        REMK_intersection_weights(length(demands), 0.0),
    ]
    grid_results = _multi_item_newsvendor_grid(
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
        grid_radii,
        demands,
        grid_weights,
        underage_costs,
        overage_costs,
    )
    for weight_index in eachindex(grid_weights), radius_index in eachindex(grid_radii)
        scalar_result =
            REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
                grid_radii[radius_index],
                demands,
                grid_weights[weight_index],
                underage_costs,
                overage_costs,
            )
        grid_result = grid_results[radius_index, weight_index]
        @test grid_result[1] ≈ scalar_result[1] atol = 1.0e-3
        @test grid_result[2] ≈ scalar_result[2] atol = 1.0e-2
    end

    extreme_ratio = 1.0e200
    extreme_epsilon = 1.0e-200
    extreme_demands = [fill(number_of_consumers / 2.0, number_of_items)]
    extreme_intersection_weights = [REMK_intersection_weights(1, extreme_ratio)]
    extreme_grid_result = _multi_item_newsvendor_grid(
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
        [extreme_epsilon],
        extreme_demands,
        extreme_intersection_weights,
        underage_costs,
        overage_costs,
    )[1, 1]
    extreme_scalar_result =
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
            extreme_epsilon,
            extreme_demands,
            extreme_intersection_weights[1],
            underage_costs,
            overage_costs,
        )
    @test extreme_grid_result[1] ≈ extreme_scalar_result[1] atol = 1.0e-3
    @test extreme_grid_result[2] ≈ extreme_scalar_result[2] atol = 1.0e-2

    # One intersection ball is exactly an ordinary W2 ball with its REMK radius.
    one_demand = [demands[1]]
    radius_ratio = 0.25
    intersection_objective, _ =
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
            1.0,
            one_demand,
            REMK_intersection_weights(1, radius_ratio),
            underage_costs,
            overage_costs,
        )
    weighted_objective, _ =
        W2_DRO_multi_item_newsvendor_objective_value_and_order(
            1.0 * (1.0 + radius_ratio), one_demand, [1.0],
            underage_costs,
            overage_costs,
        )
    @test intersection_objective ≈ weighted_objective atol = 1.0e-3

    # A zero radius ratio intentionally arrives as an all-zero REMK vector.
    zero_ratio_reference_objective = reference_intersection_W2(4.0, demands, 0.0)
    zero_ratio_objective, _ =
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
            4.0,
            demands,
            REMK_intersection_weights(length(demands), 0.0),
            underage_costs,
            overage_costs,
        )
    @test zero_ratio_objective ≈ zero_ratio_reference_objective atol = 1.0e-3

    saturated_intersection_objective, saturated_intersection_order =
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
            support_diameter,
            demands,
            REMK_intersection_weights(length(demands), 0.0),
            underage_costs,
            overage_costs,
        )
    @test saturated_intersection_objective ≈ expected_saturated_objective atol = 1.0e-10
    @test saturated_intersection_order ≈ saturated_order atol = 1.0e-10

    zero_radius_distinct_objective, zero_radius_distinct_order =
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
            0.0,
            [zeros(number_of_items), ones(number_of_items)],
            REMK_intersection_weights(2, 0.0),
            underage_costs,
            overage_costs,
        )
    @test zero_radius_distinct_objective ≈ 0.0 atol = 1.0e-10
    # Both radii are zero, so the shared increase closes the gap symmetrically
    # and first contact is the midpoint of the two centers.
    @test zero_radius_distinct_order ≈ fill(0.5, number_of_items) atol = 1.0e-8
    zero_radius_objective, zero_radius_order =
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
            0.0,
            [demands[1], copy(demands[1])],
            REMK_intersection_weights(2, 0.0),
            underage_costs,
            overage_costs,
        )
    @test zero_radius_objective ≈ 0.0
    @test zero_radius_order == demands[1]

    # For an empty intersection, keep epsilon and rho fixed and grow every
    # radius by the smallest shared increase. At rho = 0 both radii are equal,
    # so first contact is the midpoint of the two centers.
    disjoint_demands = [zeros(number_of_items), fill(number_of_consumers, number_of_items)]
    fallback_objective, fallback_order =
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
            1.0,
            disjoint_demands,
            REMK_intersection_weights(length(disjoint_demands), 0.0),
            underage_costs,
            overage_costs,
        )
    @test fallback_objective ≈ 0.0 atol = 1.0e-8
    @test fallback_order ≈ fill(0.5 * number_of_consumers, number_of_items) atol =
        1.0e-8
    ratio_fallback_objective, ratio_fallback_order =
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
            1.0,
            disjoint_demands,
            REMK_intersection_weights(length(disjoint_demands), 0.2),
            underage_costs,
            overage_costs,
        )
    @test ratio_fallback_objective ≈ fallback_objective atol = 1.0e-10
    # A positive rho makes the older ball the larger one, so first contact
    # shifts off the midpoint toward the newer, smaller ball. With
    # epsilon = 1, rho = 0.2 the radii are 1.4 and 1.2, and the contact point
    # sits at 0.5 + (r_1 - r_2) / (2 * ||d_1 - d_2||) along the segment.
    expected_ratio_coordinate =
        0.5 + 0.1 / (number_of_consumers * sqrt(number_of_items))
    @test ratio_fallback_order ≈ fill(
        number_of_consumers * expected_ratio_coordinate, number_of_items,
    ) atol = 1.0e-8

    tangent_demands = [zeros(number_of_items), [2.0, 0.0, 0.0]]
    tangent_objective, tangent_order =
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
            1.0,
            tangent_demands,
            REMK_intersection_weights(length(tangent_demands), 0.0),
            underage_costs,
            overage_costs,
        )
    @test tangent_objective ≈ 0.0 atol = 1.0e-8
    @test tangent_order ≈ [1.0, 0.0, 0.0] atol = 1.0e-3

    saa_objective, saa_order =
        SO_multi_item_newsvendor_objective_value_and_order(
            0.0, demands, weights,
            underage_costs,
            overage_costs,
        )
    @test saa_order == [4.0, 6.0, 7.0]
    @test saa_objective ≈ 4.0

    active_objective, _ =
        W2_DRO_multi_item_newsvendor_objective_value_and_order(
            1.2, demands[2:end], weights[2:end],
            underage_costs,
            overage_costs,
        )
    zero_weight_objective, _ =
        W2_DRO_multi_item_newsvendor_objective_value_and_order(
            1.2, demands, [0.0; weights[2:end]],
            underage_costs,
            overage_costs,
        )
    @test zero_weight_objective ≈ active_objective atol = 1.0e-3

    # With multiple Julia threads this executes the two solver callbacks on
    # separate, statically pinned Gurobi environments.
    threaded_objectives = zeros(2)
    Threads.@threads :static for callback_index in 1:2
        if callback_index == 1
            threaded_objectives[callback_index], _ =
                W2_DRO_multi_item_newsvendor_objective_value_and_order(
                    1.2, demands, weights,
                    underage_costs,
                    overage_costs,
                )
        else
            threaded_objectives[callback_index], _ =
                REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
                    2.5,
                    demands,
                    REMK_intersection_weights(length(demands), 0.2),
                    underage_costs,
                    overage_costs,
                )
        end
    end
    @test threaded_objectives[1] ≈ weighted_reference_objective atol = 1.0e-3
    @test threaded_objectives[2] ≈ intersection_reference_objective atol = 1.0e-3
end


@testset "intersection ball pruning" begin
    # Exact duplicates keep exactly the first occurrence.
    @test _active_intersection_ball_indices(
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], [0.3, 0.3],
    ) == [1]
    # Concentric balls keep only the smaller one.
    @test _active_intersection_ball_indices(
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], [0.3, 0.2],
    ) == [2]
    # ||d_1 - d_2|| = 0.1 sqrt(3) <= 0.5 - 0.2, so ball 1 contains ball 2.
    @test _active_intersection_ball_indices(
        [[0.6, 0.6, 0.6], [0.5, 0.5, 0.5]], [0.5, 0.2],
    ) == [2]
    # Neither ball contains the other.
    @test _active_intersection_ball_indices(
        [[0.9, 0.5, 0.5], [0.1, 0.5, 0.5]], [0.3, 0.3],
    ) == [1, 2]
    # A ball covering the whole support box is vacuous: the farthest corner
    # from (0.5, 0.5, 0.5) is at squared distance 0.75 < 0.9^2.
    @test _active_intersection_ball_indices(
        [[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]], [0.9, 0.2],
    ) == [2]
    @test isempty(
        _active_intersection_ball_indices([[0.5, 0.5, 0.5]], [2.0]),
    )

end


@testset "shared increase repair grid" begin
    repair_demands = [
        zeros(number_of_items),
        fill(number_of_consumers, number_of_items),
    ]
    repair_epsilons = [0.5, 1.0]
    # The duplicated ratio and out-of-order placement check that grid results
    # depend on parameter values rather than column order.
    repair_ratios = [0.2, 0.0, 0.1, 0.2]
    repair_weights = [REMK_intersection_weights(2, ratio) for ratio in repair_ratios]
    repair_grid = _multi_item_newsvendor_grid(
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
        repair_epsilons,
        repair_demands,
        repair_weights,
        underage_costs,
        overage_costs,
    )
    # Every cell collapses to a point mass, and the contact point sits on the
    # segment between the two centers, offset from the midpoint by half the
    # radius difference. The two balls differ by exactly epsilon * ratio.
    center_distance = sqrt(number_of_items)
    for radius_index in eachindex(repair_epsilons), weight_index in eachindex(repair_weights)
        objective, order = repair_grid[radius_index, weight_index]
        @test objective ≈ 0.0 atol = 1.0e-10
        radius_difference =
            repair_epsilons[radius_index] * repair_ratios[weight_index] /
            number_of_consumers
        expected_coordinate = 0.5 + radius_difference / (2.0 * center_distance)
        @test order ≈ fill(
            number_of_consumers * expected_coordinate, number_of_items,
        ) atol = 1.0e-8
    end

    # Equal radius profiles must give bit-identical cells regardless of column.
    for radius_index in eachindex(repair_epsilons)
        @test repair_grid[radius_index, 1][2] ≈ repair_grid[radius_index, 4][2] atol =
            1.0e-12
    end
    # A larger ratio makes the older ball larger and pushes contact away from
    # the midpoint, toward the newer center.
    for radius_index in eachindex(repair_epsilons)
        @test repair_grid[radius_index, 2][2][1] < repair_grid[radius_index, 3][2][1]
        @test repair_grid[radius_index, 3][2][1] < repair_grid[radius_index, 1][2][1]
    end

    # At the same epsilon, a sufficiently large requested rho is genuinely
    # interior while rho = 0 still needs repair. The shared contact point must
    # act as a strict-interior certificate for the former cell.
    mixed_weights = [
        REMK_intersection_weights(2, 6.0),
        REMK_intersection_weights(2, 0.0),
    ]
    mixed_grid = _multi_item_newsvendor_grid(
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
        [1.0],
        repair_demands,
        mixed_weights,
        underage_costs,
        overage_costs,
    )
    mixed_scalar =
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
            1.0, repair_demands, mixed_weights[1],
            underage_costs,
            overage_costs,
        )
    @test mixed_grid[1, 1][1] ≈ mixed_scalar[1] atol = 1.0e-3 rtol = 1.0e-4
    @test mixed_grid[1, 1][2] ≈ mixed_scalar[2] atol = 1.0e-3 rtol = 1.0e-4
    @test mixed_grid[1, 1][1] > 0.0
    @test mixed_grid[1, 2][1] ≈ 0.0 atol = 1.0e-10

    # With two centers one unit apart and epsilon = 1, the ratio that makes the
    # base radii sum to the center distance is exactly tangent, so it needs no
    # increase while the rho = 0 column still does.
    axis_demands = [zeros(number_of_items), [number_of_consumers, 0.0, 0.0]]
    tangent_ratio = 8.0 / 3.0
    tangent_grid = _multi_item_newsvendor_grid(
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
        [1.0],
        axis_demands,
        [
            REMK_intersection_weights(2, 0.0),
            REMK_intersection_weights(2, tangent_ratio),
        ],
        underage_costs,
        overage_costs,
    )
    # Equal radii: the midpoint.
    @test tangent_grid[1, 1][2] ≈ [number_of_consumers / 2.0, 0.0, 0.0] atol = 1.0e-8
    # Tangent radii 19/3 and 11/3: contact at the larger radius along the axis.
    @test tangent_grid[1, 2][2] ≈ [19.0 / 3.0, 0.0, 0.0] atol = 1.0e-8
end


@testset "intersection optimizations equivalence" begin
    # The zero-multiplier threshold agrees with the per-ball check on both
    # sides of the boundary, exercised through the grid entry point.
    threshold_demands = [demands[k] for k in 1:3]
    threshold_ratio = 0.3
    K = length(threshold_demands)
    relative_radii = [1.0 + (K - k + 1) * threshold_ratio for k in 1:K]
    normalized_threshold = _zero_multiplier_epsilon_threshold(
        [demand ./ number_of_consumers for demand in threshold_demands],
        relative_radii,
        underage_costs,
        overage_costs,
    )
    threshold_epsilon = normalized_threshold * number_of_consumers
    boundary_radii = [0.999 * threshold_epsilon, 1.001 * threshold_epsilon]
    boundary_weights = [REMK_intersection_weights(K, threshold_ratio)]
    boundary_grid = _multi_item_newsvendor_grid(
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
        boundary_radii,
        threshold_demands,
        boundary_weights,
        underage_costs,
        overage_costs,
    )
    for radius_index in eachindex(boundary_radii)
        scalar_result =
            REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
                boundary_radii[radius_index],
                threshold_demands,
                boundary_weights[1],
                underage_costs,
                overage_costs,
            )
        @test boundary_grid[radius_index, 1][1] ≈ scalar_result[1] atol = 1.0e-3
        @test boundary_grid[radius_index, 1][2] ≈ scalar_result[2] atol = 1.0e-2
    end

    # Randomized instances compared against the exhaustive oracle, both with
    # the closed-form dual solver enabled (default) and disabled (conic
    # solves with pruned multipliers), across ratios and radii that exercise
    # touching, interior, and saturated regimes.
    rng = MersenneTwister(20260710)
    for trial in 1:8
        K = rand(rng, 2:5)
        trial_demands = [
            round.(number_of_consumers .* rand(rng, number_of_items); digits = 2)
            for _ in 1:K
        ]
        trial_ratio = rand(rng, [0.0, 0.05, 0.3, 1.0])
        trial_weights = REMK_intersection_weights(K, trial_ratio)
        normalized_trial_demands =
            [demand ./ number_of_consumers for demand in trial_demands]
        trial_relative_radii = [1.0 + (K - k + 1) * trial_ratio for k in 1:K]
        minimum_epsilon, _ = reference_minimum_intersection_geometry(
            normalized_trial_demands, trial_relative_radii,
        )
        for epsilon_scale in [1.1, 2.0, 8.0]
            trial_epsilon = max(
                number_of_consumers * minimum_epsilon * epsilon_scale,
                0.05 * number_of_consumers,
            )
            oracle_objective =
                reference_intersection_W2(trial_epsilon, trial_demands, trial_ratio)

            multi_item_enable_intersection_dual_solver[] = true
            dual_objective, dual_order =
                REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
                    trial_epsilon, trial_demands, trial_weights,
                    underage_costs,
                    overage_costs,
                )
            multi_item_enable_intersection_dual_solver[] = false
            conic_objective, conic_order =
                REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
                    trial_epsilon, trial_demands, trial_weights,
                    underage_costs,
                    overage_costs,
                )
            multi_item_enable_intersection_dual_solver[] = true

            @test dual_objective ≈ oracle_objective atol = 2.0e-3 rtol = 1.0e-4
            @test conic_objective ≈ oracle_objective atol = 2.0e-3 rtol = 1.0e-4
            @test dual_objective ≈ conic_objective atol = 2.0e-3 rtol = 1.0e-4
            @test all(isapprox.(
                dual_order,
                conic_order;
                atol = 5.0e-4 * number_of_consumers,
                rtol = 5.0e-4,
            ))
            @test all((0.0 .<= dual_order) .& (dual_order .<= number_of_consumers))
            @test all((0.0 .<= conic_order) .& (conic_order .<= number_of_consumers))
        end
    end

    # Grid and single-call paths agree on a radius sweep spanning all regimes
    # for a longer history with duplicated demand points.
    sweep_demands = [
        demands[1], demands[2], copy(demands[1]), demands[3], demands[4],
        copy(demands[2]),
    ]
    sweep_K = length(sweep_demands)
    for sweep_ratio in [0.0, 0.1, 1.0]
        sweep_weights = [REMK_intersection_weights(sweep_K, sweep_ratio)]
        sweep_radii = number_of_consumers .* [1.0e-3, 0.05, 0.2, 0.5, 1.0, 3.0]
        sweep_grid = _multi_item_newsvendor_grid(
            REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
            sweep_radii,
            sweep_demands,
            sweep_weights,
            underage_costs,
            overage_costs,
        )
        for radius_index in eachindex(sweep_radii)
            scalar_result =
                REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
                    sweep_radii[radius_index], sweep_demands, sweep_weights[1],
                    underage_costs,
                    overage_costs,
                )
            @test sweep_grid[radius_index, 1][1] ≈ scalar_result[1] atol = 1.0e-3 rtol = 1.0e-4
            @test sweep_grid[radius_index, 1][2] ≈ scalar_result[2] atol = 1.0e-2 rtol = 1.0e-3
        end
    end
end
