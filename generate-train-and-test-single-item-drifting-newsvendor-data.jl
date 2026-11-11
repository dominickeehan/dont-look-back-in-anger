using Random, Statistics, StatsBase, Distributions
using ProgressBars

# Runs the single-item drifting newsvendor experiment (roughly an hour of
# compute) and writes every repetition's expected next-period cost to
# costs_file, so that plot-train-and-test-single-item-drifting-newsvendor.jl
# can redraw the figure without rerunning the experiment. Costs are flushed
# drift by drift as they are produced.

const costs_file = joinpath(
    @__DIR__,
    "single-item-drifting-newsvendor-costs.tsv",
)

const number_of_items = 1
const underage_cost_values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
const overage_cost_values = [1.0]
const number_of_consumers = 1000
const minimum_purchase_probability = 0.01
const maximum_purchase_probability = 0.99
const mixture_weight_values = [0.9, 0.95, 0.99]
const number_of_multinomials = 2

const drifts = [1.79e-3, 3.16e-3, 5.62e-3, 1.00e-2, 1.79e-2, 3.16e-2, 5.62e-2, 1.00e-1, 1.79e-1, 3.16e-1]

const number_of_repetitions = 3000
const number_of_future_samples = 1000
const history_length = 100
const training_length = 30
const global_repetition_indices = 1:number_of_repetitions

LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))

const zero_ambiguity_radius = [0.0]
const epsilon_grid = sqrt(number_of_items) * number_of_consumers * unique([
    0.0;
    LinRange(1.0e-3, 1.0e-2, 10);
    LinRange(1.0e-2, 1.0e-1, 10);
    LinRange(1.0e-1, 1.0e0, 10)
])
const smoothing_parameter_grid = [0.0; LogRange(1.0e-4, 1.0e0, 30)]
const radius_ratio_grid = [0.0; LogRange(1.0e-4, 1.0e0, 30)]
const window_size_grid = unique(round.(Int, LogRange(1, history_length, 30)))


include("weights.jl")
include("multi-item-newsvendor-dual-optimizations.jl")
include("multi-item-drifting-newsvendor-experiment-setup.jl")


smoothing_weight_vector_table = precompute_weight_vector_table(
    smoothing_weights,
    smoothing_parameter_grid,
)
SAA_weight_vector_table = precompute_weight_vector_table(
    windowing_weights,
    [history_length],
)
windowing_weight_vector_table = precompute_weight_vector_table(
    windowing_weights,
    window_size_grid,
)
intersection_weight_vector_table = precompute_weight_vector_table(
    REMK_intersection_weights,
    radius_ratio_grid,
)
weighted_W2_weight_vector_table = precompute_weight_vector_table(
    W2_linear_drift_profile_weights,
    radius_ratio_grid,
)

repetition_underage_costs = sample_repetition_item_costs(
    underage_cost_values,
    underage_cost_stream,
)
repetition_overage_costs = sample_repetition_item_costs(
    overage_cost_values,
    overage_cost_stream,
)
repetition_mixture_weights = sample_repetition_mixture_weights()

results_file = open(costs_file, "w")
try
    println(
        results_file,
        join(
            (
                "drift",
                "repetition_index",
                "method",
                "expected_next_period_cost",
            ),
            '\t',
        ),
    )

    for drift in drifts
        println("Binomial drift parameter: $drift")
        demand_sequences, final_demand_probabilities, _ =
            generate_drift_data(drift, repetition_mixture_weights)

        smoothing_costs = zeros(number_of_repetitions)
        SAA_costs = zeros(number_of_repetitions)
        windowing_costs = zeros(number_of_repetitions)
        intersection_costs = zeros(number_of_repetitions)
        weighted_costs = zeros(number_of_repetitions)

        Threads.@threads :static for repetition_index in ProgressBar(
            1:number_of_repetitions,
        )
            demand_samples = demand_sequences[repetition_index]
            instance_underage_costs =
                repetition_underage_costs[repetition_index]
            instance_overage_costs =
                repetition_overage_costs[repetition_index]
            smoothing_grid_result = train_and_test_grid_result(
                SO_multi_item_newsvendor_objective_value_and_order,
                zero_ambiguity_radius,
                smoothing_weight_vector_table,
                demand_samples,
                instance_underage_costs,
                instance_overage_costs,
            )
            SAA_grid_result = train_and_test_grid_result(
                SO_multi_item_newsvendor_objective_value_and_order,
                zero_ambiguity_radius,
                SAA_weight_vector_table,
                demand_samples,
                instance_underage_costs,
                instance_overage_costs,
            )
            windowing_grid_result = train_and_test_grid_result(
                SO_multi_item_newsvendor_objective_value_and_order,
                zero_ambiguity_radius,
                windowing_weight_vector_table,
                demand_samples,
                instance_underage_costs,
                instance_overage_costs,
            )
            intersection_grid_result = train_and_test_grid_result(
                REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
                epsilon_grid,
                intersection_weight_vector_table,
                demand_samples,
                instance_underage_costs,
                instance_overage_costs,
            )
            weighted_grid_result = train_and_test_grid_result(
                W2_DRO_multi_item_newsvendor_objective_value_and_order,
                epsilon_grid,
                weighted_W2_weight_vector_table,
                demand_samples,
                instance_underage_costs,
                instance_overage_costs,
            )

            for (costs, grid_result) in (
                (smoothing_costs, smoothing_grid_result),
                (SAA_costs, SAA_grid_result),
                (windowing_costs, windowing_grid_result),
                (intersection_costs, intersection_grid_result),
                (weighted_costs, weighted_grid_result),
            )
                _, order = grid_result[1, 1]
                costs[repetition_index] = expected_multi_item_newsvendor_cost(
                    order,
                    final_demand_probabilities[repetition_index],
                    repetition_mixture_weights[repetition_index],
                    instance_underage_costs,
                    instance_overage_costs,
                )
            end
        end

        for (method, costs) in (
            ("SAA", SAA_costs),
            ("Windowing", windowing_costs),
            ("Smoothing", smoothing_costs),
            ("Intersection", intersection_costs),
            ("Weighted", weighted_costs),
        )
            for repetition_index in 1:number_of_repetitions
                println(
                    results_file,
                    join(
                        (
                            drift,
                            repetition_index,
                            method,
                            costs[repetition_index],
                        ),
                        '\t',
                    ),
                )
            end
        end
        flush(results_file)
    end
finally
    close(results_file)
end

println("Saved per-repetition costs to $costs_file.")
