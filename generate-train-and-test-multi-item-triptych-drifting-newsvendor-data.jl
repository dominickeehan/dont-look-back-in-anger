# Each HPC array job runs the Cartesian product of ten item counts and three
# drift values. Every method is trained by rolling-origin validation and every
# hyperparameter combination is written to one triptych CSV file. Item demands
# are multinomial within a period, while their marginal distributions remain
# Binomial and therefore admit exact expected-cost tests.
#
# The demand model is the one shared with the single-item experiment. Within a
# repetition, all thirty item-count-and-drift cells are common random numbers of
# each other: the item costs, the mode weights, the mode sequence, the demand
# uniforms, the starting Gamma variates, and the unit innovations are all keyed
# by item index and reused, so a k-item instance is the k-item prefix of the
# ten-item one. Moving along the item-count axis then perturbs an instance
# instead of redrawing it, which is what makes the curves smooth.

using Random, Statistics, StatsBase, Distributions
using ProgressBars


include("experiment-randomness.jl")


const job_number = parse(Int, get(ENV, "PBS_ARRAY_INDEX", "0"))


# These bindings must exist before including the conic optimization routines.
const item_counts = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
number_of_items::Int = first(item_counts)
const number_of_consumers = 1000
const underage_cost_values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
const overage_cost_values = [1.0]
const minimum_purchase_probability = 0.01
const maximum_purchase_probability = 0.99

# A candidate first-mode weight w produces the two mode weights [w, 1-w].
const mixture_weights = [0.9, 0.95, 0.99]
const number_of_modes = 2

# Starting purchase probabilities are sampled independently for every mode.
const initial_demand_probabilities = nothing

const drifts = [0.01, 0.05, 0.1]

const number_of_repetitions = 10
const number_of_future_samples = 1000
const history_length = 100
const training_length = 30
const global_repetition_indices =
    (number_of_repetitions * job_number + 1):(
        number_of_repetitions * (job_number + 1)
    )


include("weights.jl")
include("multi-item-newsvendor-dual-optimizations.jl")

# The demand model itself is shared with the single-item experiment, so both
# draw their instances from the same streams. It supplies
# project_purchase_probabilities!, sample_multinomial_demand,
# initial_mode_demand_probabilities, generate_drift_data,
# realized_multi_item_newsvendor_cost, and the exact integer-order expected
# newsvendor cost.
include("drifting-demand-model.jl")


function sample_repetition_item_costs(cost_values, stream)
    return [
        begin
            rng = experiment_rng(global_repetition_index, stream)
            [rand(rng, cost_values) for _ in 1:number_of_items]
        end
        for global_repetition_index in global_repetition_indices
    ]
end


function sample_repetition_mode_weights()
    return [
        begin
            rng = experiment_rng(
                global_repetition_index,
                mode_weight_stream,
            )
            first_mode_weight = Float64(rand(rng, mixture_weights))
            [first_mode_weight, 1.0 - first_mode_weight]
        end
        for global_repetition_index in global_repetition_indices
    ]
end


# The expected newsvendor cost is piecewise linear in the order quantity
# because demand is integer valued, so this evaluates the exact cost at
# fractional orders too. The knot-and-interpolate routine the single-item
# script uses returns the same values; this form is kept here because the
# expected cost is evaluated at every point of the hyperparameter grid, and
# this costs two distribution-function evaluations rather than four.
function expected_newsvendor_cost_at_fractional_order(
    order,
    binomial_demand_probability,
    underage_cost,
    overage_cost,
)
    previous_trial_cdf = cdf(
        Binomial(number_of_consumers - 1, binomial_demand_probability),
        order - 1,
    )
    demand_cdf = cdf(
        Binomial(number_of_consumers, binomial_demand_probability),
        order,
    )

    expected_underage_cost = underage_cost * (
        number_of_consumers * binomial_demand_probability *
        (1.0 - previous_trial_cdf) -
        order * (1.0 - demand_cdf)
    )
    expected_overage_cost = overage_cost * (
        order * demand_cdf -
        number_of_consumers * binomial_demand_probability *
        previous_trial_cdf
    )
    return expected_underage_cost + expected_overage_cost
end


function expected_multi_item_newsvendor_cost(
    order,
    future_demand_probabilities,
    mode_weights,
    instance_underage_costs,
    instance_overage_costs,
)
    total_cost = 0.0
    inverse_future_sample_count =
        1.0 / size(future_demand_probabilities, 1)
    for future_index in axes(future_demand_probabilities, 1)
        for mode_index in axes(future_demand_probabilities, 2)
            cost_weight =
                inverse_future_sample_count * mode_weights[mode_index]
            for item_index in 1:number_of_items
                total_cost +=
                    cost_weight *
                    expected_newsvendor_cost_at_fractional_order(
                        order[item_index],
                        future_demand_probabilities[
                            future_index,
                            mode_index,
                            item_index,
                        ],
                        instance_underage_costs[item_index],
                        instance_overage_costs[item_index],
                    )
            end
        end
    end
    return total_cost
end


LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))

const window_size_grid =
    unique(round.(Int, LogRange(1, history_length, 30)))
const smoothing_parameter_grid = [0.0; LogRange(1.0e-4, 1.0e0, 30)]
const radius_ratio_grid = [0.0; LogRange(1.0e-4, 1.0e0, 30)]
const zero_ambiguity_radius = [0.0]
epsilon_grid_for_number_of_items(item_count) =
    sqrt(item_count) * number_of_consumers * unique([
        0.0;
        LinRange(1.0e-3, 1.0e-2, 10);
        LinRange(1.0e-2, 1.0e-1, 10);
        LinRange(1.0e-1, 1.0e0, 10)
    ])



serialize_vector(values) = join(values, ";")

function serialize_mode_probabilities(mode_probabilities)
    return join((serialize_vector(probabilities) for probabilities in
                 mode_probabilities), "|")
end


function precompute_weights(compute_weights, weight_parameters)
    first_sample_count = history_length - training_length
    return [
        [
            compute_weights(sample_count, weight_parameter)
            for weight_parameter in weight_parameters
        ]
        for sample_count in first_sample_count:history_length
    ]
end


function train_and_test(
    results_file,
    method,
    objective_value_and_order,
    ambiguity_radii,
    weight_parameter_name,
    weight_parameters,
    precomputed_weights,
    drift,
    demand_sequences,
    final_demand_probabilities,
    starting_demand_probabilities,
    repetition_mode_weights,
    repetition_underage_costs,
    repetition_overage_costs,
)
    first_sample_count = history_length - training_length
    println("Training and testing $method...")
    for repetition_index in 1:number_of_repetitions
        start_time = time()
        instance_underage_costs =
            repetition_underage_costs[repetition_index]
        instance_overage_costs =
            repetition_overage_costs[repetition_index]
        demand_sequence = demand_sequences[repetition_index]

        average_training_costs = zeros(
            length(ambiguity_radii),
            length(weight_parameters),
        )
        objective_values = zeros(
            length(ambiguity_radii),
            length(weight_parameters),
        )
        expected_next_period_costs = zeros(
            length(ambiguity_radii),
            length(weight_parameters),
        )
        orders = Matrix{Vector{Float64}}(
            undef,
            length(ambiguity_radii),
            length(weight_parameters),
        )

        Threads.@threads for ambiguity_radius_index in ProgressBar(eachindex(ambiguity_radii)) # Since .pbs calls for 1 thread, this is allowed on HPC.
        #for ambiguity_radius_index in ProgressBar(eachindex(ambiguity_radii))
            for weight_parameter_index in eachindex(weight_parameters)
                for time_index in
                    (first_sample_count + 1):history_length
                    sample_count = time_index - 1
                    weights = precomputed_weights[
                        sample_count - first_sample_count + 1
                    ][weight_parameter_index]
                    demand_samples = demand_sequence[1:sample_count]
                    _, order = objective_value_and_order(
                        ambiguity_radii[ambiguity_radius_index],
                        demand_samples,
                        weights,
                        instance_underage_costs,
                        instance_overage_costs,
                    )
                    average_training_costs[
                        ambiguity_radius_index,
                        weight_parameter_index,
                    ] += realized_multi_item_newsvendor_cost(
                        order,
                        demand_sequence[time_index],
                        instance_underage_costs,
                        instance_overage_costs,
                    ) / training_length
                end
            end
        end

        Threads.@threads for ambiguity_radius_index in ProgressBar(eachindex(ambiguity_radii)) # Since .pbs calls for 1 thread, this is allowed on HPC.
        #for ambiguity_radius_index in ProgressBar(eachindex(ambiguity_radii))
            for weight_parameter_index in eachindex(weight_parameters)
                weights = precomputed_weights[end][weight_parameter_index]
                objective_value, order = objective_value_and_order(
                    ambiguity_radii[ambiguity_radius_index],
                    demand_sequence,
                    weights,
                    instance_underage_costs,
                    instance_overage_costs,
                )
                objective_values[
                    ambiguity_radius_index,
                    weight_parameter_index,
                ] = objective_value
                orders[
                    ambiguity_radius_index,
                    weight_parameter_index,
                ] = order
                expected_next_period_costs[
                    ambiguity_radius_index,
                    weight_parameter_index,
                ] = expected_multi_item_newsvendor_cost(
                    order,
                    final_demand_probabilities[repetition_index],
                    repetition_mode_weights[repetition_index],
                    instance_underage_costs,
                    instance_overage_costs,
                )
            end
        end

        time_elapsed = time() - start_time
        # Match the column-major tie order used by argmin.
        for weight_parameter_index in eachindex(weight_parameters)
            for ambiguity_radius_index in eachindex(ambiguity_radii)
                output_fields = (
                    job_number,
                    number_of_items,
                    drift,
                    repetition_index,
                    method,
                    ambiguity_radii[ambiguity_radius_index],
                    weight_parameter_name,
                    weight_parameters[weight_parameter_index],
                    average_training_costs[
                        ambiguity_radius_index,
                        weight_parameter_index,
                    ],
                    objective_values[
                        ambiguity_radius_index,
                        weight_parameter_index,
                    ],
                    expected_next_period_costs[
                        ambiguity_radius_index,
                        weight_parameter_index,
                    ],
                    serialize_vector(orders[
                        ambiguity_radius_index,
                        weight_parameter_index,
                    ]),
                    serialize_vector(instance_underage_costs),
                    serialize_vector(instance_overage_costs),
                    serialize_vector(
                        repetition_mode_weights[repetition_index],
                    ),
                    serialize_mode_probabilities(
                        starting_demand_probabilities[repetition_index],
                    ),
                    time_elapsed,
                )
                println(results_file, join(output_fields, ","))
            end
        end
        flush(results_file)
    end
    return nothing
end


saa_weights = precompute_weights(windowing_weights, [history_length])
windowing_weight_table =
    precompute_weights(windowing_weights, window_size_grid)
smoothing_weight_table =
    precompute_weights(smoothing_weights, smoothing_parameter_grid)
intersection_weight_table =
    precompute_weights(REMK_intersection_weights, radius_ratio_grid)
weighted_W2_weight_table =
    precompute_weights(W2_weights, radius_ratio_grid)

results_file = open("multi-item-triptych-$job_number.csv", "w")
try
    println(
        results_file,
        join((
            "job_number",
            "number_of_items",
            "drift",
            "repetition_index",
            "method",
            "ambiguity_radius",
            "weight_parameter_name",
            "weight_parameter",
            "average_training_cost",
            "objective_value",
            "expected_next_period_cost",
            "order",
            "underage_costs",
            "overage_costs",
            "mode_weights",
            "initial_demand_probabilities",
            "time_elapsed",
        ), ","),
    )

    for item_count in item_counts
        global number_of_items = item_count
        epsilon_grid = epsilon_grid_for_number_of_items(item_count)
        repetition_underage_costs =
            sample_repetition_item_costs(
                underage_cost_values,
                underage_cost_stream,
            )
        repetition_overage_costs =
            sample_repetition_item_costs(
                overage_cost_values,
                overage_cost_stream,
            )
        repetition_mode_weights = sample_repetition_mode_weights()

        for drift in drifts
            println("Number of items: $item_count; drift: $drift")
            demand_sequences,
            final_demand_probabilities,
            starting_demand_probabilities =
                generate_drift_data(drift, repetition_mode_weights)

            train_and_test(
                results_file,
                "SAA",
                SO_multi_item_newsvendor_objective_value_and_order,
                zero_ambiguity_radius,
                "window_size",
                [history_length],
                saa_weights,
                drift,
                demand_sequences,
                final_demand_probabilities,
                starting_demand_probabilities,
                repetition_mode_weights,
                repetition_underage_costs,
                repetition_overage_costs,
            )
            train_and_test(
                results_file,
                "Windowing",
                SO_multi_item_newsvendor_objective_value_and_order,
                zero_ambiguity_radius,
                "window_size",
                window_size_grid,
                windowing_weight_table,
                drift,
                demand_sequences,
                final_demand_probabilities,
                starting_demand_probabilities,
                repetition_mode_weights,
                repetition_underage_costs,
                repetition_overage_costs,
            )
            train_and_test(
                results_file,
                "Smoothing",
                SO_multi_item_newsvendor_objective_value_and_order,
                zero_ambiguity_radius,
                "alpha",
                smoothing_parameter_grid,
                smoothing_weight_table,
                drift,
                demand_sequences,
                final_demand_probabilities,
                starting_demand_probabilities,
                repetition_mode_weights,
                repetition_underage_costs,
                repetition_overage_costs,
            )
            train_and_test(
                results_file,
                "Intersection",
                REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
                epsilon_grid,
                "rho_over_epsilon",
                radius_ratio_grid,
                intersection_weight_table,
                drift,
                demand_sequences,
                final_demand_probabilities,
                starting_demand_probabilities,
                repetition_mode_weights,
                repetition_underage_costs,
                repetition_overage_costs,
            )
            train_and_test(
                results_file,
                "Weighted",
                W2_DRO_multi_item_newsvendor_objective_value_and_order,
                epsilon_grid,
                "rho_over_epsilon",
                radius_ratio_grid,
                weighted_W2_weight_table,
                drift,
                demand_sequences,
                final_demand_probabilities,
                starting_demand_probabilities,
                repetition_mode_weights,
                repetition_underage_costs,
                repetition_overage_costs,
            )
        end
    end
finally
    close(results_file)
end
