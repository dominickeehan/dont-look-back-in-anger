using Test


number_of_items = 3
number_of_consumers = 10.0
budget = 8.0
underage_costs = fill(4.0, number_of_items)
overage_costs = fill(1.0, number_of_items)

include("weights.jl")
include("multi-item-newsvendor-optimizations.jl")


function assert_budget_feasible(order)
    @test all((0.0 .<= order) .& (order .<= number_of_consumers))
    @test sum(order) <= budget + 1.0e-6
end


@testset "multi-item order budget" begin
    demands = [[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]]
    weights = [0.5, 0.5]

    saa_objective, saa_order =
        SO_multi_item_newsvendor_objective_value_and_order(
            0.0, demands, weights,
            underage_costs,
            overage_costs,
        )
    assert_budget_feasible(saa_order)
    @test sum(saa_order) ≈ budget atol = 1.0e-7
    @test saa_objective ≈ 70.0 atol = 1.0e-7

    weighted_objective, weighted_order =
        W2_DRO_multi_item_newsvendor_objective_value_and_order(
            1.0, demands, weights,
            underage_costs,
            overage_costs,
        )
    @test weighted_objective >= saa_objective - 1.0e-6
    assert_budget_feasible(weighted_order)

    support_diameter = number_of_consumers * sqrt(number_of_items)
    saturated_objective, saturated_order =
        W2_DRO_multi_item_newsvendor_objective_value_and_order(
            support_diameter, demands, weights,
            underage_costs,
            overage_costs,
        )
    assert_budget_feasible(saturated_order)
    @test sum(saturated_order) ≈ budget atol = 1.0e-6
    @test saturated_objective ≈ 88.0 atol = 1.0e-5

    intersection_weights = REMK_intersection_weights(length(demands), 0.2)
    intersection_objective, intersection_order =
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
            2.5, demands, intersection_weights,
            underage_costs,
            overage_costs,
        )
    @test intersection_objective >= 0.0
    assert_budget_feasible(intersection_order)

    saturated_intersection_objective, saturated_intersection_order =
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
            support_diameter, demands, intersection_weights,
            underage_costs,
            overage_costs,
        )
    assert_budget_feasible(saturated_intersection_order)
    @test saturated_intersection_objective ≈ 88.0 atol = 1.0e-5

    weighted_grid = _multi_item_newsvendor_grid(
        W2_DRO_multi_item_newsvendor_objective_value_and_order,
        [0.0, 1.0],
        demands,
        [weights],
        underage_costs,
        overage_costs,
    )
    intersection_grid = _multi_item_newsvendor_grid(
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
        [2.5, support_diameter],
        demands,
        [intersection_weights],
        underage_costs,
        overage_costs,
    )
    for result in (weighted_grid..., intersection_grid...)
        assert_budget_feasible(result[2])
    end
    @test weighted_grid[1, 1][1] ≈ saa_objective atol = 1.0e-6
    @test weighted_grid[2, 1][1] ≈ weighted_objective atol = 1.0e-6
    @test intersection_grid[1, 1][1] ≈ intersection_objective atol = 1.0e-6
    @test intersection_grid[2, 1][1] ≈
        saturated_intersection_objective atol = 1.0e-6
end


@testset "budget fast-path acceptance and fallback" begin
    low_demands = [[1.0, 1.0, 1.0], [1.2, 1.2, 1.2]]
    high_demands = [[8.0, 8.0, 8.0], [8.2, 8.2, 8.2]]
    weights = [0.5, 0.5]

    # The unconstrained SAA quantiles are already under budget.
    low_saa = SO_multi_item_newsvendor_objective_value_and_order(
        0.0, low_demands, weights,
        underage_costs,
        overage_costs,
    )
    @test low_saa[2] == [1.2, 1.2, 1.2]
    assert_budget_feasible(low_saa[2])

    # A feasible weighted-W2 closed-form result is returned without numerical
    # changes from a conic solve.
    closed_form_data = _prepare_weighted_W2_closed_form(low_demands, weights, underage_costs, overage_costs)
    normalized_demands, quantiles, displacement_terms = closed_form_data
    fast_weighted = _solve_weighted_W2_closed_form(
        0.1,
        normalized_demands,
        quantiles,
        displacement_terms,
        weights,
        underage_costs,
        overage_costs,
    )
    public_weighted = W2_DRO_multi_item_newsvendor_objective_value_and_order(
        0.1, low_demands, weights,
        underage_costs,
        overage_costs,
    )
    @test public_weighted == fast_weighted
    assert_budget_feasible(public_weighted[2])
    fast_weighted_grid = _multi_item_newsvendor_grid(
        W2_DRO_multi_item_newsvendor_objective_value_and_order,
        [0.1],
        low_demands,
        [weights],
        underage_costs,
        overage_costs,
    )
    @test fast_weighted_grid[1, 1] == fast_weighted

    # The certified intersection dual is accepted when feasible.
    multi_item_reset_solver_statistics!()
    fast_intersection =
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
            0.5,
            low_demands,
            REMK_intersection_weights(2, 0.2),
            underage_costs,
            overage_costs,
        )
    fast_statistics = multi_item_solver_statistics_summary()
    assert_budget_feasible(fast_intersection[2])
    @test fast_statistics.dual_solver_solutions == 1
    @test fast_statistics.conic_solutions == 0

    multi_item_reset_solver_statistics!()
    fast_intersection_grid = _multi_item_newsvendor_grid(
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
        [0.5],
        low_demands,
        [REMK_intersection_weights(2, 0.2)],
        underage_costs,
        overage_costs,
    )
    fast_grid_statistics = multi_item_solver_statistics_summary()
    assert_budget_feasible(fast_intersection_grid[1, 1][2])
    @test fast_intersection_grid[1, 1][1] ≈ fast_intersection[1] atol = 1.0e-6
    @test fast_grid_statistics.dual_solver_solutions == 1
    @test fast_grid_statistics.conic_solutions == 0

    # The same dual path is rejected for a high-demand, over-budget order and
    # the exact coupled conic model supplies the returned solution.
    multi_item_reset_solver_statistics!()
    fallback_intersection =
        REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
            0.5,
            high_demands,
            REMK_intersection_weights(2, 0.2),
            underage_costs,
            overage_costs,
        )
    fallback_statistics = multi_item_solver_statistics_summary()
    assert_budget_feasible(fallback_intersection[2])
    @test fallback_statistics.dual_solver_solutions == 0
    @test fallback_statistics.dual_solver_failures == 0
    @test fallback_statistics.conic_solutions == 1
end
