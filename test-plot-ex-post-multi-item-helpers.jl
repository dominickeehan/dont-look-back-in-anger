using Test
using Statistics
using Distributions


# Load the real plotting helpers without running the experiment or displaying a
# plot. Preserve the caller's environment so this test is safe to include from
# a larger test process.
previous_skip_main = get(ENV, "PLOT_EX_POST_SKIP_MAIN", nothing)
ENV["PLOT_EX_POST_SKIP_MAIN"] = "1"
try
    include("plot-ex-post-multi-item-drifting-newsvendor.jl")
finally
    if isnothing(previous_skip_main)
        delete!(ENV, "PLOT_EX_POST_SKIP_MAIN")
    else
        ENV["PLOT_EX_POST_SKIP_MAIN"] = previous_skip_main
    end
end


function two_cdf_expected_cost(order, probability, consumers, underage, overage)
    previous_trial_cdf = cdf(Binomial(consumers - 1, probability), order - 1)
    demand_cdf = cdf(Binomial(consumers, probability), order)
    expected_underage = underage * (
        consumers * probability * (1.0 - previous_trial_cdf) -
        order * (1.0 - demand_cdf)
    )
    expected_overage = overage * (
        order * demand_cdf -
        consumers * probability * previous_trial_cdf
    )
    return expected_underage + expected_overage
end


function direct_expected_cost(order, probability, consumers, underage, overage)
    demand = Binomial(consumers, probability)
    return sum(
        (
            underage * max(demand_value - order, 0.0) +
            overage * max(order - demand_value, 0.0)
        ) * pdf(demand, demand_value)
        for demand_value in 0:consumers
    )
end


function direct_expected_multi_item_cost(order, final_probabilities, config)
    bounded_order = clamp.(
        order,
        0.0,
        Float64(config.consumer_count),
    )
    return mean(
        sum(
            config.mode_weights[mode_index] * direct_expected_cost(
                bounded_order[item_index],
                probabilities[mode_index][item_index],
                config.consumer_count,
                config.underage_cost,
                config.overage_cost,
            )
            for item_index in 1:config.item_count,
                mode_index in 1:config.mode_count
        )
        for probabilities in final_probabilities
    )
end


@testset "one-CDF Binomial expected-cost kernel" begin
    underage = 4.0
    overage = 1.0
    for consumers in (1, 2, 17, 1000),
        probability in (0.01, 0.37, 0.99),
        order in unique((0, 1, consumers ÷ 2, consumers - 1, consumers))
        expected = two_cdf_expected_cost(
            order,
            probability,
            consumers,
            underage,
            overage,
        )
        actual = expected_newsvendor_cost_with_binomial_demand(
            order,
            probability,
            consumers,
            underage,
            overage,
        )
        @test actual ≈ expected atol = 2.0e-12 rtol = 2.0e-12
    end

    # A direct support sum is independent of both closed-form CDF formulas.
    for probability in (0.05, 0.4, 0.95), order in (0, 1, 7, 25)
        actual = expected_newsvendor_cost_with_binomial_demand(
            order,
            probability,
            25,
            underage,
            overage,
        )
        expected = direct_expected_cost(
            order,
            probability,
            25,
            underage,
            overage,
        )
        @test actual ≈ expected atol = 2.0e-12 rtol = 2.0e-12
    end

    @test @inferred(
        expected_newsvendor_cost_with_binomial_demand(
            4,
            0.3,
            10,
            underage,
            overage,
        ),
    ) isa Float64
    @test_throws MethodError expected_newsvendor_cost_with_binomial_demand(
        4.5,
        0.3,
        10,
        underage,
        overage,
    )
    @test_throws ArgumentError expected_newsvendor_cost_with_binomial_demand(
        -1,
        0.3,
        10,
        underage,
        overage,
    )
    @test_throws ArgumentError expected_newsvendor_cost_with_binomial_demand(
        11,
        0.3,
        10,
        underage,
        overage,
    )
end


@testset "shared dense order-knot lookup" begin
    config = ExPostExperimentConfig(
        2,
        2,
        [0.25, 0.75],
        [[0.2, 0.4], [0.7, 0.8]],
        [0.01],
        12,
        4.0,
        1.0,
        1,
        1,
        2,
        42,
    )
    final_probabilities = [
        [[0.15, 0.40], [0.65, 0.75]],
        [[0.20, 0.35], [0.70, 0.80]],
    ]

    result_type = Tuple{Float64,Vector{Float64},Int}
    first_method = Matrix{result_type}(undef, 2, 1)
    first_method[1, 1] = (0.0, [0.0, 1.25], 0)
    first_method[2, 1] = (0.0, [2.2, 12.0], 0)

    second_method = Matrix{result_type}(undef, 1, 2)
    second_method[1, 1] = (0.0, [5.75, 3.4], 0)
    # Exercise both support clamps as well as knots disjoint from method one.
    second_method[1, 2] = (0.0, [13.5, -1.0], 0)

    shared_costs = precompute_expected_costs_at_order_knots(
        (first_method, second_method),
        final_probabilities,
        config,
    )
    first_method_costs = precompute_expected_costs_at_order_knots(
        (first_method,),
        final_probabilities,
        config,
    )
    second_method_costs = precompute_expected_costs_at_order_knots(
        (second_method,),
        final_probabilities,
        config,
    )

    expected_knots = (
        Set((0, 2, 3, 5, 6, 12)),
        Set((0, 1, 2, 3, 4, 12)),
    )
    for item_index in 1:config.item_count, order in 0:config.consumer_count
        @test isfinite(shared_costs[item_index][order + 1]) ==
            (order in expected_knots[item_index])
    end

    for (method, isolated_costs) in (
        (first_method, first_method_costs),
        (second_method, second_method_costs),
    )
        for result in method
            order = result[2]
            shared_result = expected_multi_item_cost_from_order_knots(
                order,
                shared_costs,
                config,
            )
            isolated_result = expected_multi_item_cost_from_order_knots(
                order,
                isolated_costs,
                config,
            )
            direct_result = direct_expected_multi_item_cost(
                order,
                final_probabilities,
                config,
            )
            @test shared_result ≈ isolated_result atol = 2.0e-12 rtol = 2.0e-12
            @test shared_result ≈ direct_result atol = 2.0e-12 rtol = 2.0e-12
        end
    end
end
