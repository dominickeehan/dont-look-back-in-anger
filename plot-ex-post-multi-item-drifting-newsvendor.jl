using Random, Statistics, StatsBase, Distributions
using ProgressBars


# These bindings must exist before including the optimization routines, which
# copy them into their own typed constants.
const number_of_items = 1
const number_of_modes = 2
const mixture_weights = [0.9, 0.1]
const initial_demand_probabilities = [
    0.1 * ones(number_of_items),
    0.5 * ones(number_of_items),
]
construct_drift_distribution(delta) = TriangularDist(-delta, delta, 0.0)
const drifts = [1.00e-2, 3.16e-2, 1.00e-1, 3.16e-1, 1.00e0]
const number_of_consumers = 1000
const cu = 4.0 # Per-unit underage cost.
const co = 1.0 # Per-unit overage cost.

include("weights.jl")
include("multi-item-newsvendor-optimizations.jl")

const number_of_repetitions = parse(
    Int,
    get(ENV, "PLOT_EX_POST_REPETITIONS", "10000"),
)
const number_of_future_samples = parse(
    Int,
    get(ENV, "PLOT_EX_POST_FUTURE_SAMPLES", "100"),
)
const history_length = 100
const simulation_seed = 42


struct ExPostExperimentConfig
    item_count::Int
    mode_count::Int
    mode_weights::Vector{Float64}
    initial_probabilities::Vector{Vector{Float64}}
    drift_values::Vector{Float64}
    consumer_count::Int
    underage_cost::Float64
    overage_cost::Float64
    repetition_count::Int
    history_length::Int
    future_sample_count::Int
    seed::Int
end


const experiment_config = ExPostExperimentConfig(
    number_of_items,
    number_of_modes,
    mixture_weights,
    initial_demand_probabilities,
    drifts,
    number_of_consumers,
    cu,
    co,
    number_of_repetitions,
    history_length,
    number_of_future_samples,
    simulation_seed,
)


# For D ~ Binomial(n, p),
#   F_{n-1}(q-1) = F_n(q) - (n-q)/n * P(D=q).
# This replaces the second Binomial CDF in the original expression with an
# allocation-free PDF evaluation while preserving the exact expected cost.
@inline function expected_newsvendor_cost_with_binomial_demand(
    order::Int,
    binomial_demand_probability::Float64,
    consumer_count::Int,
    underage_cost::Float64,
    overage_cost::Float64,
)::Float64
    0 <= order <= consumer_count ||
        throw(ArgumentError("order must lie between zero and consumer_count"))
    demand_distribution = Binomial(
        consumer_count,
        binomial_demand_probability,
    )
    demand_cdf = cdf(demand_distribution, order)
    previous_trial_cdf = clamp(
        demand_cdf -
        ((consumer_count - order) / consumer_count) *
        pdf(demand_distribution, order),
        0.0,
        1.0,
    )

    expected_underage_cost = underage_cost * (
        consumer_count * binomial_demand_probability *
        (1.0 - previous_trial_cdf) -
        order * (1.0 - demand_cdf)
    )
    expected_overage_cost = overage_cost * (
        order * demand_cdf -
        consumer_count * binomial_demand_probability * previous_trial_cdf
    )
    return expected_underage_cost + expected_overage_cost
end


function _mark_order_knots!(requested_orders, grid_results, config)
    for result in grid_results
        order = result[2]
        for item_index in 1:config.item_count
            bounded_order = clamp(
                order[item_index],
                0.0,
                Float64(config.consumer_count),
            )
            requested_orders[item_index][floor(Int, bounded_order) + 1] = true
            requested_orders[item_index][ceil(Int, bounded_order) + 1] = true
        end
    end
    return nothing
end


# All four methods use the same simulated future distributions. Build one
# dense lookup from the union of their integer order knots, rather than four
# separate Set/Dict tables and four repeated future-cost passes.
function precompute_expected_costs_at_order_knots(
    method_grid_results::Tuple,
    final_demand_probabilities::Vector{Vector{Vector{Float64}}},
    config::ExPostExperimentConfig,
)
    requested_orders = [
        falses(config.consumer_count + 1)
        for _ in 1:config.item_count
    ]
    for grid_results in method_grid_results
        _mark_order_knots!(requested_orders, grid_results, config)
    end

    expected_costs = [
        fill(NaN, config.consumer_count + 1)
        for _ in 1:config.item_count
    ]
    inverse_future_sample_count = 1.0 / length(final_demand_probabilities)
    for item_index in 1:config.item_count
        for order_storage_index in eachindex(requested_orders[item_index])
            requested_orders[item_index][order_storage_index] || continue
            integer_order = order_storage_index - 1
            total_cost = 0.0
            for demand_probabilities in final_demand_probabilities
                for mode_index in 1:config.mode_count
                    total_cost +=
                        config.mode_weights[mode_index] *
                        expected_newsvendor_cost_with_binomial_demand(
                            integer_order,
                            demand_probabilities[mode_index][item_index],
                            config.consumer_count,
                            config.underage_cost,
                            config.overage_cost,
                        )
                end
            end
            expected_costs[item_index][order_storage_index] =
                total_cost * inverse_future_sample_count
        end
    end
    return expected_costs
end


function expected_multi_item_cost_from_order_knots(
    order,
    expected_costs::Vector{Vector{Float64}},
    config::ExPostExperimentConfig,
)::Float64
    total_cost = 0.0
    for item_index in 1:config.item_count
        bounded_order = clamp(
            order[item_index],
            0.0,
            Float64(config.consumer_count),
        )
        lower_order = floor(Int, bounded_order)
        upper_order = ceil(Int, bounded_order)
        lower_cost = expected_costs[item_index][lower_order + 1]
        if lower_order == upper_order
            total_cost += lower_cost
        else
            interpolation_weight = bounded_order - lower_order
            total_cost +=
                (1.0 - interpolation_weight) * lower_cost +
                interpolation_weight *
                expected_costs[item_index][upper_order + 1]
        end
    end
    return total_cost
end


function generate_drift_data(
    config::ExPostExperimentConfig,
    drift::Float64,
)
    Random.seed!(config.seed)
    drift_distribution = construct_drift_distribution(drift)
    mode_sampler = Weights(config.mode_weights)

    demand_sequences = Vector{Vector{Vector{Float64}}}(
        undef,
        config.repetition_count,
    )
    final_demand_probabilities =
        Vector{Vector{Vector{Vector{Float64}}}}(
            undef,
            config.repetition_count,
        )

    for repetition_index in 1:config.repetition_count
        demand_probabilities = deepcopy(config.initial_probabilities)
        demand_sequence = Vector{Vector{Float64}}(
            undef,
            config.history_length,
        )
        future_probabilities = Vector{Vector{Vector{Float64}}}(
            undef,
            config.future_sample_count,
        )

        for time_index in 1:config.history_length
            mode = sample(1:config.mode_count, mode_sampler)
            demand_sequence[time_index] = [
                Float64(rand(Binomial(
                    config.consumer_count,
                    demand_probabilities[mode][item_index],
                )))
                for item_index in 1:config.item_count
            ]

            if time_index < config.history_length
                demand_probabilities = [
                    [
                        clamp(
                            demand_probabilities[mode_index][item_index] +
                            rand(drift_distribution),
                            0.01,
                            0.99,
                        )
                        for item_index in 1:config.item_count
                    ]
                    for mode_index in 1:config.mode_count
                ]
            else
                for future_index in 1:config.future_sample_count
                    future_probabilities[future_index] = [
                        [
                            clamp(
                                demand_probabilities[mode_index][item_index] +
                                rand(drift_distribution),
                                0.01,
                                0.99,
                            )
                            for item_index in 1:config.item_count
                        ]
                        for mode_index in 1:config.mode_count
                    ]
                end
            end
        end

        demand_sequences[repetition_index] = demand_sequence
        final_demand_probabilities[repetition_index] = future_probabilities
    end
    return demand_sequences, final_demand_probabilities
end


LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))

const zero_ambiguity_radius = [0.0]
const epsilon_grid = sqrt(number_of_items) * number_of_consumers * unique([
    0.0;
    LinRange(1.0e-3, 1.0e-2, 10);
    LinRange(1.0e-2, 1.0e-1, 10);
    LinRange(1.0e-1, 1.0e0, 10)
])
const window_size_grid = unique(round.(
    Int,
    LogRange(1, history_length, 30),
))
const smoothing_parameter_grid = [0.0; LogRange(1.0e-4, 1.0e0, 30)]
const radius_ratio_grid = [0.0; LogRange(1.0e-4, 1.0e0, 30)]
const intersection_epsilon_grid =
    sqrt(number_of_items) * number_of_consumers * unique([
        LinRange(1.0e-3, 1.0e-2, 10);
        LinRange(1.0e-2, 1.0e-1, 10);
        LinRange(1.0e-1, 1.0e0, 10)
    ])
const intersection_radius_ratio_grid = [
    0.0;
    LogRange(1.0e-4, 1.0e0, 30)
]


function precompute_weight_vectors(compute_weights, parameters, history_length)
    weight_vectors = Vector{Vector{Float64}}(undef, length(parameters))
    Threads.@threads for parameter_index in eachindex(parameters)
        weight_vectors[parameter_index] = compute_weights(
            history_length,
            parameters[parameter_index],
        )
    end
    return weight_vectors
end


function _fill_ex_post_costs!(
    costs,
    doubling_counts,
    repetition_index,
    grid_results,
    expected_costs,
    config,
)
    for weight_parameter_index in axes(grid_results, 2),
        ambiguity_radius_index in axes(grid_results, 1)
        _, order, count =
            grid_results[ambiguity_radius_index, weight_parameter_index]
        costs[
            ambiguity_radius_index,
            weight_parameter_index,
            repetition_index,
        ] = expected_multi_item_cost_from_order_knots(
            order,
            expected_costs,
            config,
        )
        doubling_counts[
            ambiguity_radius_index,
            weight_parameter_index,
            repetition_index,
        ] = count
    end
    return nothing
end


function _allocate_method_storage(
    ambiguity_radii,
    weight_parameters,
    repetition_count,
)
    dimensions = (
        length(ambiguity_radii),
        length(weight_parameters),
        repetition_count,
    )
    return zeros(Float64, dimensions), zeros(Int, dimensions)
end


function _summarize_method!(
    average_costs,
    standard_deviations,
    drift_index,
    method_name,
    costs,
    doubling_counts,
    ambiguity_radii,
    weight_parameters,
)
    mean_costs = dropdims(mean(costs; dims = 3); dims = 3)
    ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean_costs))
    minimal_costs = view(
        costs,
        ambiguity_radius_index,
        weight_parameter_index,
        :,
    )

    digits = 4
    average_cost = round(mean(minimal_costs); digits = digits)
    standard_deviation = round(sem(minimal_costs); digits = digits)
    optimal_ambiguity_radius = round(
        ambiguity_radii[ambiguity_radius_index];
        digits = digits,
    )
    optimal_weight_parameter = round(
        weight_parameters[weight_parameter_index];
        digits = digits,
    )
    optimal_doubling_count = round(
        mean(view(
            doubling_counts,
            ambiguity_radius_index,
            weight_parameter_index,
            :,
        ));
        digits = digits,
    )

    println(method_name)
    print("Ex-post minimal average cost: $average_cost ± $standard_deviation, ")
    print("Optimal ambiguity radius: $optimal_ambiguity_radius, ")
    print("Weight parameter: $optimal_weight_parameter, ")
    println("Doubling count: $optimal_doubling_count")

    average_costs[drift_index] = average_cost
    standard_deviations[drift_index] = standard_deviation
    return nothing
end


function _display_and_reset_solver_statistics!()
    solver_statistics = multi_item_solver_statistics_summary()
    if solver_statistics.touching_solutions +
       solver_statistics.zero_multiplier_solutions +
       solver_statistics.single_ball_solutions +
       solver_statistics.dual_solver_solutions +
       solver_statistics.conic_solutions +
       solver_statistics.numeric_retry_solves +
       solver_statistics.pair_certificate_solutions > 0
        println(solver_statistics)
    end
    multi_item_reset_solver_statistics!()
    return nothing
end


# Process every method for a repetition before discarding its temporary grids.
# The simulated histories and dense expected-cost lookup are consequently
# shared by smoothing, windowing, intersection, and weighted W2.
function compute_ex_post_lines(config::ExPostExperimentConfig)
    smoothing_weight_vectors = precompute_weight_vectors(
        smoothing_weights,
        smoothing_parameter_grid,
        config.history_length,
    )
    windowing_weight_vectors = precompute_weight_vectors(
        windowing_weights,
        window_size_grid,
        config.history_length,
    )
    intersection_weight_vectors = precompute_weight_vectors(
        REMK_intersection_weights,
        intersection_radius_ratio_grid,
        config.history_length,
    )
    weighted_W2_weight_vectors = precompute_weight_vectors(
        W2_weights,
        radius_ratio_grid,
        config.history_length,
    )

    drift_count = length(config.drift_values)
    smoothing_average_costs = zeros(drift_count)
    smoothing_standard_deviations = zeros(drift_count)
    windowing_average_costs = zeros(drift_count)
    windowing_standard_deviations = zeros(drift_count)
    intersection_average_costs = zeros(drift_count)
    intersection_standard_deviations = zeros(drift_count)
    weighted_average_costs = zeros(drift_count)
    weighted_standard_deviations = zeros(drift_count)

    for drift_index in eachindex(config.drift_values)
        drift = config.drift_values[drift_index]
        println("Binomial drift parameter: $drift")
        demand_sequences, final_demand_probabilities =
            generate_drift_data(config, drift)

        smoothing_costs, smoothing_doubling_counts = _allocate_method_storage(
            zero_ambiguity_radius,
            smoothing_parameter_grid,
            config.repetition_count,
        )
        windowing_costs, windowing_doubling_counts = _allocate_method_storage(
            zero_ambiguity_radius,
            window_size_grid,
            config.repetition_count,
        )
        intersection_costs, intersection_doubling_counts =
            _allocate_method_storage(
                intersection_epsilon_grid,
                intersection_radius_ratio_grid,
                config.repetition_count,
            )
        weighted_costs, weighted_doubling_counts = _allocate_method_storage(
            epsilon_grid,
            radius_ratio_grid,
            config.repetition_count,
        )

        multi_item_reset_solver_statistics!()
        Threads.@threads :static for repetition_index in ProgressBar(
            1:config.repetition_count,
        )
            demand_samples = demand_sequences[repetition_index]
            smoothing_grid_results = _multi_item_newsvendor_grid(
                SO_multi_item_newsvendor_objective_value_and_order,
                zero_ambiguity_radius,
                demand_samples,
                smoothing_weight_vectors,
                0,
            )
            windowing_grid_results = _multi_item_newsvendor_grid(
                SO_multi_item_newsvendor_objective_value_and_order,
                zero_ambiguity_radius,
                demand_samples,
                windowing_weight_vectors,
                0,
            )
            intersection_grid_results = _multi_item_newsvendor_grid(
                REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
                intersection_epsilon_grid,
                demand_samples,
                intersection_weight_vectors,
                0,
            )
            weighted_grid_results = _multi_item_newsvendor_grid(
                W2_DRO_multi_item_newsvendor_objective_value_and_order,
                epsilon_grid,
                demand_samples,
                weighted_W2_weight_vectors,
                0,
            )

            method_grid_results = (
                smoothing_grid_results,
                windowing_grid_results,
                intersection_grid_results,
                weighted_grid_results,
            )
            expected_costs = precompute_expected_costs_at_order_knots(
                method_grid_results,
                final_demand_probabilities[repetition_index],
                config,
            )

            _fill_ex_post_costs!(
                smoothing_costs,
                smoothing_doubling_counts,
                repetition_index,
                smoothing_grid_results,
                expected_costs,
                config,
            )
            _fill_ex_post_costs!(
                windowing_costs,
                windowing_doubling_counts,
                repetition_index,
                windowing_grid_results,
                expected_costs,
                config,
            )
            _fill_ex_post_costs!(
                intersection_costs,
                intersection_doubling_counts,
                repetition_index,
                intersection_grid_results,
                expected_costs,
                config,
            )
            _fill_ex_post_costs!(
                weighted_costs,
                weighted_doubling_counts,
                repetition_index,
                weighted_grid_results,
                expected_costs,
                config,
            )
        end

        _display_and_reset_solver_statistics!()
        _summarize_method!(
            smoothing_average_costs,
            smoothing_standard_deviations,
            drift_index,
            "Smoothing",
            smoothing_costs,
            smoothing_doubling_counts,
            zero_ambiguity_radius,
            smoothing_parameter_grid,
        )
        _summarize_method!(
            windowing_average_costs,
            windowing_standard_deviations,
            drift_index,
            "Windowing",
            windowing_costs,
            windowing_doubling_counts,
            zero_ambiguity_radius,
            window_size_grid,
        )
        _summarize_method!(
            intersection_average_costs,
            intersection_standard_deviations,
            drift_index,
            "Intersection",
            intersection_costs,
            intersection_doubling_counts,
            intersection_epsilon_grid,
            intersection_radius_ratio_grid,
        )
        _summarize_method!(
            weighted_average_costs,
            weighted_standard_deviations,
            drift_index,
            "Weighted",
            weighted_costs,
            weighted_doubling_counts,
            epsilon_grid,
            radius_ratio_grid,
        )
    end

    return (
        smoothing = (
            average_costs = smoothing_average_costs,
            standard_deviations = smoothing_standard_deviations,
        ),
        windowing = (
            average_costs = windowing_average_costs,
            standard_deviations = windowing_standard_deviations,
        ),
        intersection = (
            average_costs = intersection_average_costs,
            standard_deviations = intersection_standard_deviations,
        ),
        weighted = (
            average_costs = weighted_average_costs,
            standard_deviations = weighted_standard_deviations,
        ),
    )
end


using Plots, Measures


function plot_ex_post_lines(results, config::ExPostExperimentConfig)
    default() # Reset plot defaults.
    gr(size = (275 + 6 + 8, 183 + 6) .* sqrt(3))

    fontfamily = "Computer Modern"
    default(
        framestyle = :box,
        grid = true,
        gridalpha = 0.075,
        minorgrid = true,
        minorgridalpha = 0.075,
        minorgridlinestyle = :dash,
        tick_direction = :in,
        xminorticks = 9,
        yminorticks = 0,
        fontfamily = fontfamily,
        guidefont = Plots.font(fontfamily; pointsize = 12),
        legendfont = Plots.font(fontfamily; pointsize = 11),
        tickfont = Plots.font(fontfamily; pointsize = 10),
    )

    plt = plot(
        xscale = :log10,
        xlabel = "Binomial drift parameter, \$δ\$",
        ylabel = "Ex-post optimal expected\ncost (relative to smoothing)",
        topmargin = 0.0pt,
        leftmargin = 6.0pt,
        bottommargin = 6.0pt,
        rightmargin = 0.0pt,
        legend = :bottomleft,
    )

    fillalpha = 0.1
    normalizer = results.smoothing.average_costs

    plot!(
        plt,
        config.drift_values,
        results.windowing.average_costs ./ normalizer;
        ribbon = results.windowing.standard_deviations ./ normalizer,
        fillalpha = fillalpha,
        color = palette(:tab10)[7],
        linestyle = :dashdot,
        markershape = :pentagon,
        markersize = 4.0,
        markerstrokewidth = 0.0,
        label = "Windowing (\$ε=0\$)",
    )
    plot!(
        plt,
        config.drift_values,
        results.smoothing.average_costs ./ normalizer;
        ribbon = results.smoothing.standard_deviations ./ normalizer,
        fillalpha = fillalpha,
        color = palette(:tab10)[9],
        linestyle = :dot,
        linewidth = 1.2,
        markershape = :star4,
        markersize = 6.0,
        markerstrokewidth = 0.0,
        label = "Smoothing (\$ε=0\$)",
    )
    plot!(
        plt,
        config.drift_values,
        results.intersection.average_costs ./ normalizer;
        ribbon = results.intersection.standard_deviations ./ normalizer,
        fillalpha = fillalpha,
        color = palette(:tab10)[1],
        linestyle = :solid,
        markershape = :circle,
        markersize = 4.0,
        markerstrokewidth = 0.0,
        label = "Intersection",
    )
    plot!(
        plt,
        config.drift_values,
        results.weighted.average_costs ./ normalizer;
        ribbon = results.weighted.standard_deviations ./ normalizer,
        fillalpha = fillalpha,
        color = palette(:tab10)[2],
        linestyle = :dash,
        markershape = :diamond,
        markersize = 4.0,
        markerstrokewidth = 0.0,
        label = "Weighted",
    )
    return plt
end


function main()
    if Threads.nthreads() == 1
        @warn(
            "This experiment is running with one Julia thread. " *
            "Restart with `julia --threads=auto " *
            "plot-ex-post-multi-item-drifting-newsvendor.jl` for parallel repetitions.",
        )
    end
    results = compute_ex_post_lines(experiment_config)
    plt = plot_ex_post_lines(results, experiment_config)
    display(plt)
    return results, plt
end


if get(ENV, "PLOT_EX_POST_SKIP_MAIN", "0") != "1"
    main()
end
