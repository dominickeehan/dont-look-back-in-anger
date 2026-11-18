using Random, Statistics, StatsBase, Distributions

# Generates one job's worth of data for the process-train-and-test-multi-item-triptych-drifting-newsvendor-data.jl script
# 
# Expected to be run on an HPC (see the .pbs script).

const job_number = parse(Int, get(ENV, "PBS_ARRAY_INDEX", "0")) # Index of this job in the .pbs job array (0 when run locally).


# Newsvendor problems and their drifting demand distributions (see multi-item-drifting-newsvendor-experiment-setup.jl).
const item_counts = 1:10
number_of_items::Int = 1 # Set to each item count by the loop below.
const underage_cost_values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
const overage_cost_values = [1.0]
const number_of_multinomials = 2
const mixture_weight_values = [0.9, 0.95, 0.99]
const number_of_consumers = 1000
const minimum_purchase_probability = 0.01
const maximum_purchase_probability = 0.99
const drifts = [0.01, 0.05, 0.1]

# Repetitions run by this job. Each method is trained on the last training_length periods of a
# history of history_length periods and tested on number_of_future_samples samples of the next period.
const number_of_repetitions = 10
const repetitions = number_of_repetitions*job_number .+ (1:number_of_repetitions)
const history_length = 100
const training_length = 30
const number_of_future_samples = 1000

# Grids of weight parameters and ambiguity radii searched in training.
LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))
const window_size_grid = unique(round.(Int, LogRange(1, history_length, 30)))
const smoothing_parameter_grid = [0.0; LogRange(1.0e-4, 1.0e0, 30)]
const ambiguity_radius_ratio_grid = [0.0; LogRange(1.0e-4, 1.0e0, 30)]
ambiguity_radius_grid(item_count) = sqrt(item_count) * number_of_consumers * unique([0.0; LinRange(1.0e-3, 1.0e-2, 10); LinRange(1.0e-2, 1.0e-1, 10); LinRange(1.0e-1, 1.0e0, 10)])


include("weights.jl")
include("multi-item-newsvendor-dual-optimizations.jl")
include("multi-item-drifting-newsvendor-experiment-setup.jl")



# Methods to train and test.
methods = [("SAA", SO_multi_item_newsvendor_objective_value_and_order, false, "window_size", [history_length], weight_vector_table(windowing_weights, [history_length])),
           ("Windowing", SO_multi_item_newsvendor_objective_value_and_order, false, "window_size", window_size_grid, weight_vector_table(windowing_weights, window_size_grid)),
           ("Smoothing", SO_multi_item_newsvendor_objective_value_and_order, false, "alpha", smoothing_parameter_grid, weight_vector_table(smoothing_weights, smoothing_parameter_grid)),
           ("Intersection", REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order, true, "rho_over_epsilon", ambiguity_radius_ratio_grid, weight_vector_table(REMK_intersection_weights, ambiguity_radius_ratio_grid)),
           ("Weighted", W2_DRO_multi_item_newsvendor_objective_value_and_order, true, "rho_over_epsilon", ambiguity_radius_ratio_grid, weight_vector_table(W2_linear_drift_profile_weights, ambiguity_radius_ratio_grid))]

# Results file with one row per run.
results_file = open("multi-item-triptych-$job_number.csv", "w")
println(results_file, "job_number,number_of_items,drift,repetition_index,method,ambiguity_radius,weight_parameter_name,weight_parameter,average_training_cost,objective_value,expected_next_period_cost,order,underage_costs,overage_costs,mixture_weights,initial_demand_probabilities,time_elapsed")


# Loop through every item count and drift and save the results of each repetition and method.
for item_count in item_counts, drift in drifts

        global number_of_items = item_count
        println("Number of items: $item_count; drift: $drift")

        for (repetition_index, repetition) in enumerate(repetitions)

                # Costs and mixture weights of this repetition.
                underage_costs = sample_item_costs(repetition, underage_cost_values, underage_cost_stream)
                overage_costs = sample_item_costs(repetition, overage_cost_values, overage_cost_stream)
                mixture_weights = sample_mixture_weights(repetition)

                # Demand history and purchase probabilities of this repetition.
                demands, next_purchase_probabilities, initial_purchase_probabilities = generate_drift_data(drift, repetition, mixture_weights)

                for (method, objective_value_and_order, has_ambiguity_radius, weight_parameter_name, weight_parameters, weight_vectors) in methods

                        start_time = time()

                        # Average training cost of every ambiguity radius and weight parameter.
                        ambiguity_radii = has_ambiguity_radius ? ambiguity_radius_grid(item_count) : [0.0]
                        training_costs = average_training_costs(objective_value_and_order, ambiguity_radii, weight_vectors, demands, underage_costs, overage_costs)

                        # Order and expected next-period cost of every ambiguity radius and weight parameter.
                        rows = []
                        for (weight_parameter_index, weight_parameter) in enumerate(weight_parameters), (ambiguity_radius_index, ambiguity_radius) in enumerate(ambiguity_radii)

                                objective_value, order = objective_value_and_order(ambiguity_radius, demands, weight_vectors[history_length][weight_parameter_index], underage_costs, overage_costs)
                                expected_next_period_cost = expected_multi_item_newsvendor_cost(order, next_purchase_probabilities, mixture_weights, underage_costs, overage_costs)

                                push!(rows, (job_number, item_count, drift, repetition_index, method, ambiguity_radius, weight_parameter_name, weight_parameter,
                                             training_costs[ambiguity_radius_index,weight_parameter_index], objective_value, expected_next_period_cost,
                                             join(order, ";"), join(underage_costs, ";"), join(overage_costs, ";"), join(mixture_weights, ";"),
                                             join(join.(initial_purchase_probabilities, ";"), "|")))

                        end

                        # Save every row with the time taken by the whole method.
                        time_elapsed = time() - start_time
                        for row in rows; println(results_file, join((row..., time_elapsed), ",")); end
                        flush(results_file)

                end
        end
end

close(results_file)
