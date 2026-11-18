using Random, Statistics, StatsBase, Distributions
using ProgressBars

# Generates the data for the plot-train-and-test-single-item-drifting-newsvendor.jl script.


# Newsvendor problems and their drifting demand distributions (see multi-item-drifting-newsvendor-experiment-setup.jl).
const number_of_items = 1
const underage_cost_values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
const overage_cost_values = [1.0]
const number_of_multinomials = 2
const mixture_weight_values = [0.9, 0.95, 0.99]
const number_of_consumers = 1000
const minimum_purchase_probability = 0.01
const maximum_purchase_probability = 0.99
const drifts = [1.79e-3, 3.16e-3, 5.62e-3, 1.00e-2, 1.79e-2, 3.16e-2, 5.62e-2, 1.00e-1, 1.79e-1, 3.16e-1]

# Repetitions for each drift. Each method is trained on the last training_length periods of a
# history of history_length periods and tested on number_of_future_samples samples of the next period.
const number_of_repetitions = 3000
const history_length = 100
const training_length = 30
const number_of_future_samples = 1000

# Grids of weight parameters and ambiguity radii searched in training.
LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))
const window_size_grid = unique(round.(Int, LogRange(1, history_length, 30)))
const smoothing_parameter_grid = [0.0; LogRange(1.0e-4, 1.0e0, 30)]
const ambiguity_radius_ratio_grid = [0.0; LogRange(1.0e-4, 1.0e0, 30)]
const ambiguity_radius_grid = sqrt(number_of_items) * number_of_consumers * unique([0.0; LinRange(1.0e-3, 1.0e-2, 10); LinRange(1.0e-2, 1.0e-1, 10); LinRange(1.0e-1, 1.0e0, 10)])


include("weights.jl")
include("multi-item-newsvendor-dual-optimizations.jl")
include("multi-item-drifting-newsvendor-experiment-setup.jl")



# Methods to train and test.
methods = [("SAA", SO_multi_item_newsvendor_objective_value_and_order, [0.0], weight_vector_table(windowing_weights, [history_length])),
           ("Windowing", SO_multi_item_newsvendor_objective_value_and_order, [0.0], weight_vector_table(windowing_weights, window_size_grid)),
           ("Smoothing", SO_multi_item_newsvendor_objective_value_and_order, [0.0], weight_vector_table(smoothing_weights, smoothing_parameter_grid)),
           ("Intersection", REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order, ambiguity_radius_grid, weight_vector_table(REMK_intersection_weights, ambiguity_radius_ratio_grid)),
           ("Weighted", W2_DRO_multi_item_newsvendor_objective_value_and_order, ambiguity_radius_grid, weight_vector_table(W2_linear_drift_profile_weights, ambiguity_radius_ratio_grid))]

# Results file with one row per repetition and method for each drift.
results_file = open("single-item-drifting-newsvendor-costs.tsv", "w")
println(results_file, join(("drift", "repetition_index", "method", "expected_next_period_cost"), '\t'))


# Loop through every drift and save the expected next-period cost of each repetition and method.
for drift in drifts

        println("Drift: $drift")
        costs = zeros(number_of_repetitions, length(methods))

        Threads.@threads for repetition in ProgressBar(1:number_of_repetitions)

                # Costs and mixture weights of this repetition.
                underage_costs = sample_item_costs(repetition, underage_cost_values, underage_cost_stream)
                overage_costs = sample_item_costs(repetition, overage_cost_values, overage_cost_stream)
                mixture_weights = sample_mixture_weights(repetition)

                # Demand history and purchase probabilities of this repetition.
                demands, next_purchase_probabilities, _ = generate_drift_data(drift, repetition, mixture_weights)

                # Trained order and its expected next-period cost for every method.
                for (method_index, (_, objective_value_and_order, ambiguity_radii, weight_vectors)) in enumerate(methods)

                        order = train_and_test_order(objective_value_and_order, ambiguity_radii, weight_vectors, demands, underage_costs, overage_costs)

                        costs[repetition,method_index] = expected_multi_item_newsvendor_cost(order, next_purchase_probabilities, mixture_weights, underage_costs, overage_costs)

                end
        end

        # Save the costs once all the parallel repetitions are done.
        for (method_index, (method, _)) in enumerate(methods), repetition in 1:number_of_repetitions
                println(results_file, join((drift, repetition, method, costs[repetition,method_index]), '\t'))

        end
        flush(results_file)

end

close(results_file)
