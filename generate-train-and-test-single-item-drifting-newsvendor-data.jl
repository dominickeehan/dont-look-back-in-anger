# Each HPC array job simulates one multi-item, multi-modal drifting-newsvendor
# instance for one drift value. Every method is trained by rolling-origin
# validation and every hyperparameter combination is written to the job's CSV
# file. Item demands are multinomial within a period, while their marginal
# distributions remain Binomial and therefore admit exact expected-cost tests.

using Random, Statistics, StatsBase, Distributions
using ProgressBars


# Seed from the full array index before making any random draws so every HPC
# job receives a distinct random stream.
const job_number = parse(Int, get(ENV, "PBS_ARRAY_INDEX", "0"))
Random.seed!(job_number)


# These bindings must exist before including the conic optimization routines.
const number_of_items = 1
const number_of_consumers = 1000
const underage_cost_values = [3.0, 4.0, 5.0, 6.0]
const overage_cost_values = [1.0]
const minimum_purchase_probability = 0.01
const maximum_purchase_probability = 0.99

# A candidate first-mode weight w produces the two mode weights [w, 1-w].
const mixture_weights = [0.9, 0.95, 0.99]
const number_of_modes = 2

# For one item, set this to [p_mode_1, p_mode_2] to fix the two modes'
# starting purchase probabilities. Leave it as nothing to sample them.
const initial_demand_probabilities = nothing

construct_drift_distribution(delta) = TriangularDist(-delta, delta, 0.0)
const drifts = [
    3.16e-3,    
    5.62e-3,
    1.00e-2,
    1.79e-2,
    3.16e-2,
    5.62e-2,
    1.00e-1,
    1.79e-1,
    3.16e-1,
    5.62e-1,
    1.00e-0,
]

const number_of_repetitions = 1
const number_of_future_samples = 1000
const history_length = 100
const training_length = 30


include("weights.jl")
include("multi-item-newsvendor-conic-optimizations.jl")


function project_purchase_probabilities!(purchase_probabilities)
    maximum_probability_sum = 1.0
    box_projection = clamp.(
        purchase_probabilities,
        minimum_purchase_probability,
        maximum_purchase_probability,
    )
    if sum(box_projection) <= maximum_probability_sum
        purchase_probabilities .= box_projection
        return purchase_probabilities
    end

    # When the sum constraint binds, its multiplier is the scalar shift in
    # clamp.(purchase_probabilities .- shift, lower_bound, upper_bound).
    lower_shift = 0.0
    upper_shift = maximum(
        purchase_probabilities .- minimum_purchase_probability,
    )
    for _ in 1:100
        shift = (lower_shift + upper_shift) / 2.0
        projected_sum = sum(
            clamp(
                probability - shift,
                minimum_purchase_probability,
                maximum_purchase_probability,
            ) for probability in purchase_probabilities
        )
        if projected_sum > maximum_probability_sum
            lower_shift = shift
        else
            upper_shift = shift
        end
    end

    purchase_probabilities .= clamp.(
        purchase_probabilities .- upper_shift,
        minimum_purchase_probability,
        maximum_purchase_probability,
    )
    return purchase_probabilities
end


function sample_multinomial_demand(purchase_probabilities)
    category_probabilities = vcat(
        purchase_probabilities,
        1.0 - sum(purchase_probabilities),
    )
    category_counts = rand(Multinomial(
        number_of_consumers,
        category_probabilities,
    ))
    return Float64.(category_counts[1:number_of_items])
end


function initial_mode_demand_probabilities()
    if isnothing(initial_demand_probabilities)
        return [
            project_purchase_probabilities!(
                rand(Dirichlet(number_of_items + 1, 1.0))[1:number_of_items],
            ) for _ in 1:number_of_modes
        ]
    end
    return [
        [Float64(probability)]
        for probability in initial_demand_probabilities
    ]
end


function sample_repetition_item_costs(cost_values)
    return [
        [rand(cost_values) for _ in 1:number_of_items]
        for _ in 1:number_of_repetitions
    ]
end


function sample_repetition_mode_weights()
    return [
        begin
            first_mode_weight = Float64(rand(mixture_weights))
            [first_mode_weight, 1.0 - first_mode_weight]
        end
        for _ in 1:number_of_repetitions
    ]
end


function generate_drift_data(drift, repetition_mode_weights)
    drift_distribution = construct_drift_distribution(drift)
    demand_sequences = Vector{Vector{Vector{Float64}}}(
        undef,
        number_of_repetitions,
    )
    final_demand_probabilities = Vector{Array{Float64,3}}(
        undef,
        number_of_repetitions,
    )
    starting_demand_probabilities = Vector{Vector{Vector{Float64}}}(
        undef,
        number_of_repetitions,
    )

    for repetition_index in 1:number_of_repetitions
        mode_weights = repetition_mode_weights[repetition_index]
        mode_sampler = Weights(mode_weights)
        demand_probabilities = initial_mode_demand_probabilities()
        starting_demand_probabilities[repetition_index] =
            deepcopy(demand_probabilities)
        demand_sequence = Vector{Vector{Float64}}(
            undef,
            history_length,
        )
        future_probabilities = Array{Float64}(
            undef,
            number_of_future_samples,
            number_of_modes,
            number_of_items,
        )

        for time_index in 1:history_length
            mode = sample(1:number_of_modes, mode_sampler)
            demand_sequence[time_index] =
                sample_multinomial_demand(demand_probabilities[mode])

            time_index == history_length && continue
            for mode_index in 1:number_of_modes
                mode_probabilities = demand_probabilities[mode_index]
                for item_index in eachindex(mode_probabilities)
                    mode_probabilities[item_index] += rand(drift_distribution)
                end
                project_purchase_probabilities!(mode_probabilities)
            end
        end

        # Each future sample is a separate one-step drift from the distribution
        # at the end of the observed history.
        for future_index in 1:number_of_future_samples
            for mode_index in 1:number_of_modes
                for item_index in 1:number_of_items
                    future_probabilities[
                        future_index,
                        mode_index,
                        item_index,
                    ] =
                        demand_probabilities[mode_index][item_index] +
                        rand(drift_distribution)
                end
                project_purchase_probabilities!(
                    view(future_probabilities, future_index, mode_index, :),
                )
            end
        end

        demand_sequences[repetition_index] = demand_sequence
        final_demand_probabilities[repetition_index] = future_probabilities
    end
    return (
        demand_sequences,
        final_demand_probabilities,
        starting_demand_probabilities,
    )
end


function realized_multi_item_newsvendor_cost(
    order,
    demand,
    instance_underage_costs,
    instance_overage_costs,
)
    total_cost = 0.0
    for item_index in 1:number_of_items
        total_cost +=
            instance_underage_costs[item_index] *
            max(demand[item_index] - order[item_index], 0.0) +
            instance_overage_costs[item_index] *
            max(order[item_index] - demand[item_index], 0.0)
    end
    return total_cost
end


function expected_newsvendor_cost_with_binomial_demand(
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
                    expected_newsvendor_cost_with_binomial_demand(
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
const epsilon_grid = sqrt(number_of_items) * number_of_consumers * unique([
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

        # Threads.@threads for ambiguity_radius_index in ProgressBar(eachindex(ambiguity_radii))
        for ambiguity_radius_index in ProgressBar(eachindex(ambiguity_radii))
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

        # Threads.@threads for ambiguity_radius_index in ProgressBar(eachindex(ambiguity_radii))
        for ambiguity_radius_index in ProgressBar(eachindex(ambiguity_radii))
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
        for ambiguity_radius_index in eachindex(ambiguity_radii)
            for weight_parameter_index in eachindex(weight_parameters)
                output_fields = (
                    job_number,
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


repetition_underage_costs =
    sample_repetition_item_costs(underage_cost_values)
repetition_overage_costs =
    sample_repetition_item_costs(overage_cost_values)
repetition_mode_weights = sample_repetition_mode_weights()
saa_weights = precompute_weights(windowing_weights, [history_length])
windowing_weight_table =
    precompute_weights(windowing_weights, window_size_grid)
smoothing_weight_table =
    precompute_weights(smoothing_weights, smoothing_parameter_grid)
intersection_weight_table =
    precompute_weights(REMK_intersection_weights, radius_ratio_grid)
weighted_W2_weight_table =
    precompute_weights(W2_weights, radius_ratio_grid)

results_file = open("single-item-$job_number.csv", "w")
try
    println(
        results_file,
        join((
            "job_number",
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

    for drift in drifts
        println("Drift: $drift")
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
finally
    close(results_file)
end
