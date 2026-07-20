# Demand is a mixture of two Binomial distributions for one item. Every
# repetition starts the two modes at purchase probabilities 0.1 and 0.5,
# respectively. The mixture weights and underage and overage costs retain the
# sampling scheme used by the multi-item version. Each mode's purchase
# probability then drifts independently over time.

using Random, Statistics, StatsBase, Distributions
using ProgressBars


# These bindings must exist before including the optimization routines, which
# copy them into their own typed constants.
const number_of_items = 1
const number_of_consumers = 1000
const underage_cost_values = [4.0]
const overage_cost_values = [1.0]
const minimum_purchase_probability = 0.01
const maximum_purchase_probability = 0.99

const first_mode_weight_values = [0.9] #[0.9, 0.95, 0.99]
const number_of_modes = 2
const initial_demand_probabilities = [0.1, 0.5]
construct_drift_distribution(delta) = TriangularDist(-delta, delta, 0.0)
#const drifts = [1.00e-1, 3.16e-1, 1.00e0]
const drifts = [5.62e-3, 1.00e-2, 3.16e-2, 1.00e-1, 3.16e-1, 1.00e0]
#const drifts = [5.62e-3, 1.00e-2, 1.79e-2, 3.16e-2, 5.62e-2, 1.00e-1, 1.79e-1, 3.16e-1, 5.62e-1, 1.00e0]

include("weights.jl")
include("multi-item-newsvendor-optimizations.jl")

const number_of_repetitions = 1000
const number_of_future_samples = 1000
const history_length = 100
const training_length = 30
const simulation_seed = 42


# For D ~ Binomial(n, p),
#   F_{n-1}(q-1) = F_n(q) - (n-q)/n * P(D=q).
# This replaces the second Binomial CDF in the original expression with an
# allocation-free PDF evaluation while preserving the exact expected cost.
function expected_newsvendor_cost_with_binomial_demand(
    order,
    binomial_demand_probability,
    consumer_count,
    underage_cost,
    overage_cost,
)
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


function _mark_order_knots!(requested_orders, grid_results)
    for result in grid_results
        order = result[2]
        for item_index in 1:number_of_items
            requested_orders[item_index][floor(Int, order[item_index]) + 1] = true
            requested_orders[item_index][ceil(Int, order[item_index]) + 1] = true
        end
    end
    return nothing
end


# All five methods use the same simulated future distributions. Build one
# lookup from the union of their integer order knots.
function precompute_expected_costs_at_order_knots(
    method_grid_results,
    final_demand_probabilities,
    mode_weights,
    instance_underage_costs,
    instance_overage_costs,
)
    requested_orders = [
        falses(number_of_consumers + 1)
        for _ in 1:number_of_items
    ]
    for grid_results in method_grid_results
        _mark_order_knots!(requested_orders, grid_results)
    end

    expected_costs = [
        fill(NaN, number_of_consumers + 1)
        for _ in 1:number_of_items
    ]
    inverse_future_sample_count = 1.0 / length(final_demand_probabilities)
    for item_index in 1:number_of_items
        for order_storage_index in eachindex(requested_orders[item_index])
            requested_orders[item_index][order_storage_index] || continue
            integer_order = order_storage_index - 1
            total_cost = 0.0
            for demand_probabilities in final_demand_probabilities
                for mode_index in 1:number_of_modes
                    total_cost +=
                        mode_weights[mode_index] *
                        expected_newsvendor_cost_with_binomial_demand(
                            integer_order,
                            demand_probabilities[mode_index][item_index],
                            number_of_consumers,
                            instance_underage_costs[item_index],
                            instance_overage_costs[item_index],
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
    expected_costs,
)
    total_cost = 0.0
    for item_index in 1:number_of_items
        lower_order = floor(Int, order[item_index])
        upper_order = ceil(Int, order[item_index])
        lower_cost = expected_costs[item_index][lower_order + 1]
        if lower_order == upper_order
            total_cost += lower_cost
        else
            interpolation_weight = order[item_index] - lower_order
            total_cost +=
                (1.0 - interpolation_weight) * lower_cost +
                interpolation_weight *
                expected_costs[item_index][upper_order + 1]
        end
    end
    return total_cost
end


function sample_repetition_underage_costs()
    cost_rng = MersenneTwister(simulation_seed + 1)
    return [
        [
            rand(cost_rng, underage_cost_values)
            for _ in 1:number_of_items
        ]
        for _ in 1:number_of_repetitions
    ]
end


function sample_repetition_overage_costs()
    cost_rng = MersenneTwister(simulation_seed + 2)
    return [
        [
            rand(cost_rng, overage_cost_values)
            for _ in 1:number_of_items
        ]
        for _ in 1:number_of_repetitions
    ]
end


function sample_repetition_mixture_weights()
    weight_rng = MersenneTwister(simulation_seed + 3)
    return [
        begin
            first_mode_weight = rand(weight_rng, first_mode_weight_values)
            [first_mode_weight, 1.0 - first_mode_weight]
        end
        for _ in 1:number_of_repetitions
    ]
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


function sample_binomial_demand(purchase_probability)
    demand = rand(Binomial(number_of_consumers, purchase_probability))
    return [Float64(demand)]
end


function generate_drift_data(drift, repetition_mixture_weights)
    Random.seed!(simulation_seed)
    drift_distribution = construct_drift_distribution(drift)

    demand_sequences = Vector{Vector{Vector{Float64}}}(
        undef,
        number_of_repetitions,
    )
    final_demand_probabilities =
        Vector{Vector{Vector{Vector{Float64}}}}(
            undef,
            number_of_repetitions,
        )
    for repetition_index in 1:number_of_repetitions
        mode_sampler = Weights(
            repetition_mixture_weights[repetition_index],
        )
        demand_probabilities = copy(initial_demand_probabilities)
        demand_sequence = Vector{Vector{Float64}}(
            undef,
            history_length,
        )
        future_probabilities = Vector{Vector{Vector{Float64}}}(
            undef,
            number_of_future_samples,
        )

        for time_index in 1:history_length
            mode = sample(1:number_of_modes, mode_sampler)
            demand_sequence[time_index] =
                sample_binomial_demand(demand_probabilities[mode])

            time_index == history_length && continue
            for mode_index in 1:number_of_modes
                demand_probabilities[mode_index] = clamp(
                    demand_probabilities[mode_index] +
                    rand(drift_distribution),
                    minimum_purchase_probability,
                    maximum_purchase_probability,
                )
            end
        end

        for future_index in 1:number_of_future_samples
            future_probabilities[future_index] = [
                [clamp(
                    demand_probabilities[mode_index] +
                    rand(drift_distribution),
                    minimum_purchase_probability,
                    maximum_purchase_probability,
                )]
                for mode_index in 1:number_of_modes
            ]
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
const smoothing_parameter_grid = [0.0; LogRange(1.0e-4, 1.0e0, 30)]
const radius_ratio_grid = [0.0; LogRange(1.0e-4, 1.0e0, 30)]
const window_size_grid = unique(round.(Int, LogRange(1, history_length, 30)))


function precompute_weight_vector_table(compute_weights, parameters)
    return [
        [
            compute_weights(sample_count, parameter)
            for parameter in parameters
        ]
        for sample_count in
            (history_length - training_length):history_length
    ]
end


# Select hyperparameters by rolling-origin validation, then refit the winner on
# the full history. The final one-cell grid has the same shape as an ex-post
# grid result, so all methods can share the future-cost lookup below.
function _train_and_test_grid_result(
    objective_value_and_order,
    ambiguity_radii,
    weight_vector_table,
    demand_sequence,
    instance_underage_costs,
    instance_overage_costs,
)
    first_sample_count = history_length - training_length
    training_costs = zeros(
        length(ambiguity_radii),
        length(weight_vector_table[1]),
    )

    for time_index in (first_sample_count + 1):history_length
        sample_count = time_index - 1
        weight_vectors = weight_vector_table[
            sample_count - first_sample_count + 1
        ]
        grid_results = _multi_item_newsvendor_grid(
            objective_value_and_order,
            ambiguity_radii,
            demand_sequence[1:sample_count],
            weight_vectors,
            instance_underage_costs,
            instance_overage_costs,
        )
        realized_demand = demand_sequence[time_index]
        for weight_parameter_index in axes(grid_results, 2),
            ambiguity_radius_index in axes(grid_results, 1)
            training_costs[ambiguity_radius_index, weight_parameter_index] +=
                realized_multi_item_newsvendor_cost(
                    grid_results[
                        ambiguity_radius_index,
                        weight_parameter_index,
                    ][2],
                    realized_demand,
                    instance_underage_costs,
                    instance_overage_costs,
                )
        end
    end

    ambiguity_radius_index, weight_parameter_index =
        Tuple(argmin(training_costs))
    return _multi_item_newsvendor_grid(
        objective_value_and_order,
        ambiguity_radii[
            ambiguity_radius_index:ambiguity_radius_index,
        ],
        demand_sequence,
        weight_vector_table[end][
            weight_parameter_index:weight_parameter_index
        ],
        instance_underage_costs,
        instance_overage_costs,
    )
end


function _fill_train_and_test_cost!(
    costs,
    repetition_index,
    grid_result,
    expected_costs,
)
    _, order = grid_result[1, 1]
    costs[repetition_index] = expected_multi_item_cost_from_order_knots(
        order,
        expected_costs,
    )
    return nothing
end


summarize_method(costs) = mean(costs), sem(costs)


# Process every method for a repetition so they share the same history and
# future-demand samples.
function compute_train_and_test_lines()
    smoothing_weight_vector_table = precompute_weight_vector_table(
        smoothing_weights,
        smoothing_parameter_grid,
    )
    saa_weight_vector_table = precompute_weight_vector_table(
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
        W2_weights,
        radius_ratio_grid,
    )

    drift_count = length(drifts)
    smoothing_average_costs = zeros(drift_count)
    smoothing_standard_errors = zeros(drift_count)
    saa_average_costs = zeros(drift_count)
    saa_standard_errors = zeros(drift_count)
    windowing_average_costs = zeros(drift_count)
    windowing_standard_errors = zeros(drift_count)
    intersection_average_costs = zeros(drift_count)
    intersection_standard_errors = zeros(drift_count)
    weighted_average_costs = zeros(drift_count)
    weighted_standard_errors = zeros(drift_count)
    repetition_underage_costs = sample_repetition_underage_costs()
    repetition_overage_costs = sample_repetition_overage_costs()
    repetition_mixture_weights = sample_repetition_mixture_weights()

    for drift_index in eachindex(drifts)
        drift = drifts[drift_index]
        println("Binomial drift parameter: $drift")
        demand_sequences, final_demand_probabilities =
            generate_drift_data(drift, repetition_mixture_weights)

        smoothing_costs = zeros(number_of_repetitions)
        saa_costs = zeros(number_of_repetitions)
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
            smoothing_grid_result = _train_and_test_grid_result(
                SO_multi_item_newsvendor_objective_value_and_order,
                zero_ambiguity_radius,
                smoothing_weight_vector_table,
                demand_samples,
                instance_underage_costs,
                instance_overage_costs,
            )
            saa_grid_result = _train_and_test_grid_result(
                SO_multi_item_newsvendor_objective_value_and_order,
                zero_ambiguity_radius,
                saa_weight_vector_table,
                demand_samples,
                instance_underage_costs,
                instance_overage_costs,
            )
            windowing_grid_result = _train_and_test_grid_result(
                SO_multi_item_newsvendor_objective_value_and_order,
                zero_ambiguity_radius,
                windowing_weight_vector_table,
                demand_samples,
                instance_underage_costs,
                instance_overage_costs,
            )
            intersection_grid_result = _train_and_test_grid_result(
                REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
                epsilon_grid,
                intersection_weight_vector_table,
                demand_samples,
                instance_underage_costs,
                instance_overage_costs,
            )
            weighted_grid_result = _train_and_test_grid_result(
                W2_DRO_multi_item_newsvendor_objective_value_and_order,
                epsilon_grid,
                weighted_W2_weight_vector_table,
                demand_samples,
                instance_underage_costs,
                instance_overage_costs,
            )

            method_grid_results = (
                smoothing_grid_result,
                saa_grid_result,
                windowing_grid_result,
                intersection_grid_result,
                weighted_grid_result,
            )
            expected_costs = precompute_expected_costs_at_order_knots(
                method_grid_results,
                final_demand_probabilities[repetition_index],
                repetition_mixture_weights[repetition_index],
                instance_underage_costs,
                instance_overage_costs,
            )

            _fill_train_and_test_cost!(
                smoothing_costs,
                repetition_index,
                smoothing_grid_result,
                expected_costs,
            )
            _fill_train_and_test_cost!(
                saa_costs,
                repetition_index,
                saa_grid_result,
                expected_costs,
            )
            _fill_train_and_test_cost!(
                windowing_costs,
                repetition_index,
                windowing_grid_result,
                expected_costs,
            )
            _fill_train_and_test_cost!(
                intersection_costs,
                repetition_index,
                intersection_grid_result,
                expected_costs,
            )
            _fill_train_and_test_cost!(
                weighted_costs,
                repetition_index,
                weighted_grid_result,
                expected_costs,
            )
        end

        (smoothing_average_costs[drift_index],
         smoothing_standard_errors[drift_index]) =
            summarize_method(smoothing_costs)
        (saa_average_costs[drift_index],
         saa_standard_errors[drift_index]) =
            summarize_method(saa_costs)
        (windowing_average_costs[drift_index],
         windowing_standard_errors[drift_index]) =
            summarize_method(windowing_costs)
        (intersection_average_costs[drift_index],
         intersection_standard_errors[drift_index]) =
            summarize_method(intersection_costs)
        (weighted_average_costs[drift_index],
         weighted_standard_errors[drift_index]) =
            summarize_method(weighted_costs)
    end

    return (
        smoothing = (
            average_costs = smoothing_average_costs,
            standard_errors = smoothing_standard_errors,
        ),
        saa = (
            average_costs = saa_average_costs,
            standard_errors = saa_standard_errors,
        ),
        windowing = (
            average_costs = windowing_average_costs,
            standard_errors = windowing_standard_errors,
        ),
        intersection = (
            average_costs = intersection_average_costs,
            standard_errors = intersection_standard_errors,
        ),
        weighted = (
            average_costs = weighted_average_costs,
            standard_errors = weighted_standard_errors,
        ),
    )
end


# Run the experiment when this script is loaded.
results = compute_train_and_test_lines()


using Plots, Measures

default() # Reset plot defaults.
gr(size = (275 + 6 + 8 + 3, 183 + 6 + 10) .* sqrt(3))

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
    ylabel = "Average train-and-test next-period\nexpected cost (relative to smoothing)",
    topmargin = 10.0pt,
    leftmargin = 6.0pt,
    bottommargin = 6.0pt,
    rightmargin = 3.0pt,
)

fillalpha = 0.1
normalizer = results.smoothing.average_costs

plot!(
    plt,
    drifts,
    results.saa.average_costs ./ normalizer;
    ribbon = results.saa.standard_errors ./ normalizer,
    fillalpha = fillalpha,
    color = palette(:tab10)[8],
    linestyle = :solid,
    label = "SAA",
)
plot!(
    plt,
    drifts,
    results.windowing.average_costs ./ normalizer;
    ribbon = results.windowing.standard_errors ./ normalizer,
    fillalpha = fillalpha,
    color = palette(:tab10)[7],
    linestyle = :dashdot,
    markershape = :pentagon,
    markersize = 4.0,
    markerstrokewidth = 0.0,
    label = "Windowing",
)
plot!(
    plt,
    drifts,
    results.smoothing.average_costs ./ normalizer;
    ribbon = results.smoothing.standard_errors ./ normalizer,
    fillalpha = fillalpha,
    color = palette(:tab10)[9],
    linestyle = :dot,
    linewidth = 1.2,
    markershape = :star4,
    markersize = 6.0,
    markerstrokewidth = 0.0,
    label = "Smoothing",
)
plot!(
    plt,
    drifts,
    results.intersection.average_costs ./ normalizer;
    ribbon = results.intersection.standard_errors ./ normalizer,
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
    drifts,
    results.weighted.average_costs ./ normalizer;
    ribbon = results.weighted.standard_errors ./ normalizer,
    fillalpha = fillalpha,
    color = palette(:tab10)[2],
    linestyle = :dash,
    markershape = :diamond,
    markersize = 4.0,
    markerstrokewidth = 0.0,
    label = "Weighted",
)
xticks!([1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e0])
xlims!((0.99999 * first(drifts), 1.00001 * last(drifts)))
yticks!([0.8, 0.90, 1.00, 1.10, 1.20, 1.30])
ylims!((0.79999, 1.30001))
display(plt)
