using Test
using LinearAlgebra
using Random


number_of_items = 2
number_of_consumers = 10.0
cu = 4.0
co = 1.0

include("weights.jl")
include("multi-item-newsvendor-optimizations.jl")


# Reference geometry solve that always uses the conic model, bypassing the
# planar weighted one-center solver.
function reference_minimum_intersection_epsilon(normalized_demands, relative_radii)
    problem = _new_multi_item_model()
    @variables(problem, begin
        1.0 >= feasible_point[i = 1:multi_item_dimension] >= 0.0
        minimum_normalized_epsilon >= 0.0
    end)
    for k in eachindex(normalized_demands)
        @constraint(
            problem,
            [
                relative_radii[k] * minimum_normalized_epsilon;
                [
                    feasible_point[i] - normalized_demands[k][i]
                    for i in 1:multi_item_dimension
                ]
            ] in MathOptInterface.SecondOrderCone(multi_item_dimension + 1),
        )
    end
    @objective(problem, Min, minimum_normalized_epsilon)
    _optimize_multi_item_model!(problem; high_precision = true)
    return value(minimum_normalized_epsilon)
end


@testset "planar maximizing-pair certificate" begin
    radius_ratio = 0.4
    K = 4
    relative_radii = [
        1.0 + (K - k + 1) * radius_ratio
        for k in 1:K
    ]
    first_center = [0.1, 0.25]
    last_center = [0.9, 0.75]
    pair_point = first_center .+ (
        relative_radii[1] / (relative_radii[1] + relative_radii[end])
    ) .* (last_center .- first_center)
    normalized_demands = [
        first_center,
        copy(pair_point),
        copy(pair_point),
        last_center,
    ]
    reference_epsilon =
        reference_minimum_intersection_epsilon(
            normalized_demands, relative_radii,
        )

    multi_item_reset_solver_statistics!()
    minimum_epsilon, feasible_point =
        _compute_minimum_intersection_epsilon_and_point(
            normalized_demands, radius_ratio,
        )
    statistics = multi_item_solver_statistics_summary()

    @test minimum_epsilon ≈ reference_epsilon atol = 1.0e-8 rtol = 1.0e-8
    @test feasible_point ≈ pair_point atol = 1.0e-12 rtol = 1.0e-12
    @test statistics.geometry_solves == 1
    @test statistics.pair_certificate_solutions == 1
    @test statistics.geometry_socp_solves == 0
end


@testset "planar weighted one-center geometry" begin
    rng = MersenneTwister(31415)
    for trial in 1:40
        K = rand(rng, 1:12)
        normalized_demands = [rand(rng, number_of_items) for _ in 1:K]
        if trial % 4 == 0 && K >= 2
            # Include exact duplicates and near-collinear centers.
            normalized_demands[end] = copy(normalized_demands[1])
        end
        radius_ratio = rand(rng, [0.0, 1.0e-3, 0.05, 0.4, 2.0])
        relative_radii = [1.0 + (K - k + 1) * radius_ratio for k in 1:K]

        minimum_epsilon, feasible_point =
            _compute_minimum_intersection_epsilon_and_point(
                normalized_demands, radius_ratio,
            )
        reference_epsilon =
            reference_minimum_intersection_epsilon(normalized_demands, relative_radii)

        @test minimum_epsilon ≈ reference_epsilon atol = 1.0e-7 rtol = 1.0e-7
        # The returned point must lie in the box and cover every ball at the
        # returned epsilon.
        @test all(value -> -1.0e-12 <= value <= 1.0 + 1.0e-12, feasible_point)
        coverage = maximum(
            sqrt(_squared_euclidean_distance(feasible_point, normalized_demands[k])) /
            relative_radii[k] for k in 1:K
        )
        @test coverage <= minimum_epsilon * (1.0 + 1.0e-9) + 1.0e-12
    end
end


# Exhaustive oracle at two items (four joint loss pieces), matching the
# construction in test-multi-item-newsvendor-optimizations.jl.
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


@testset "near-touching intersection conic precision" begin
    near_touching_demands = [
        [7.811483592838464, 7.536146746992138],
        [6.692330784030012, 8.820580885728184],
        [5.132242409590599, 5.550403047994981],
        [0.14636341476581327, 2.682608067600465],
        [4.922399380236504, 7.52719905184051],
        [4.582665084530417, 4.147450719888999],
    ]
    # This is 1.000001 times the minimum feasible radius. At a 1.0e-5
    # barrier convergence tolerance Gurobi reports an optimal solution here
    # whose objective is over four times too large.
    near_touching_epsilon = 4.539296020422841
    old_dual_setting = multi_item_enable_intersection_dual_solver[]
    try
        multi_item_enable_intersection_dual_solver[] = false
        objective, order, _ =
            REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
                near_touching_epsilon,
                near_touching_demands,
                REMK_intersection_weights(length(near_touching_demands), 0.0),
                0,
            )
        @test objective ≈ 0.01815811948688406 atol = 5.0e-5 rtol = 1.0e-3
        @test all(isapprox.(
            order,
            [3.8937101049406317, 5.252738536595071];
            atol = 5.0e-4,
            rtol = 1.0e-4,
        ))
    finally
        multi_item_enable_intersection_dual_solver[] = old_dual_setting
    end
end


@testset "two-item intersection equivalence" begin
    rng = MersenneTwister(2718)
    for trial in 1:8
        K = rand(rng, 2:6)
        trial_demands = [
            round.(number_of_consumers .* rand(rng, number_of_items); digits = 2)
            for _ in 1:K
        ]
        trial_ratio = rand(rng, [0.0, 0.05, 0.3, 1.0])
        trial_weights = REMK_intersection_weights(K, trial_ratio)
        normalized_trial_demands =
            [demand ./ number_of_consumers for demand in trial_demands]
        minimum_epsilon, _ = _compute_minimum_intersection_epsilon_and_point(
            normalized_trial_demands, trial_ratio,
        )
        for epsilon_scale in [1.1, 2.0, 8.0]
            trial_epsilon = max(
                number_of_consumers * minimum_epsilon * epsilon_scale,
                0.05 * number_of_consumers,
            )
            oracle_objective =
                reference_intersection_W2(trial_epsilon, trial_demands, trial_ratio)

            multi_item_enable_intersection_dual_solver[] = true
            dual_objective, dual_order, _ =
                REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
                    trial_epsilon, trial_demands, trial_weights, 0,
                )
            multi_item_enable_intersection_dual_solver[] = false
            conic_objective, conic_order, _ =
                REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
                    trial_epsilon, trial_demands, trial_weights, 0,
                )
            multi_item_enable_intersection_dual_solver[] = true

            @test dual_objective ≈ oracle_objective atol = 2.0e-3 rtol = 1.0e-4
            @test conic_objective ≈ oracle_objective atol = 2.0e-3 rtol = 1.0e-4
            @test all(isapprox.(
                dual_order,
                conic_order;
                atol = 5.0e-4 * number_of_consumers,
                rtol = 5.0e-4,
            ))
            @test all((0.0 .<= dual_order) .& (dual_order .<= number_of_consumers))
        end
    end

    # A long-history grid sweep at plot-like scale agrees with per-point calls.
    rng_grid = MersenneTwister(999)
    sweep_demands = [
        Float64.(round.(number_of_consumers .* rand(rng_grid, number_of_items); digits = 1))
        for _ in 1:30
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
            0,
        )
        for radius_index in eachindex(sweep_radii)
            scalar_result =
                REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(
                    sweep_radii[radius_index], sweep_demands, sweep_weights[1], 0,
                )
            @test sweep_grid[radius_index, 1][1] ≈ scalar_result[1] atol = 1.0e-3 rtol = 1.0e-4
            @test sweep_grid[radius_index, 1][2] ≈ scalar_result[2] atol = 1.0e-2 rtol = 1.0e-3
        end
    end
end
