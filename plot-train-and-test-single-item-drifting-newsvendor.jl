using Random, Statistics, StatsBase, Distributions
using ProgressBars


# These must be defined before including the optimization and shared
# experiment setup routines.
const number_of_items = 1
const number_of_consumers = 1000
const underage_cost_values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
const overage_cost_values = [1.0]
const minimum_purchase_probability = 0.01
const maximum_purchase_probability = 0.99
const mixture_weight_values = [0.9, 0.95, 0.99]
const number_of_multinomials = 2
const initial_demand_probabilities = nothing

#const drifts = [1.79e-1, 3.16e-1]
const drifts = [1.79e-3, 3.16e-3, 5.62e-3, 1.00e-2, 1.79e-2, 3.16e-2, 5.62e-2, 1.00e-1, 1.79e-1, 3.16e-1]

const number_of_repetitions = 2000
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

    drift_count = length(drifts)
    smoothing_average_costs = zeros(drift_count)
    smoothing_standard_errors = zeros(drift_count)
    SAA_average_costs = zeros(drift_count)
    SAA_standard_errors = zeros(drift_count)
    windowing_average_costs = zeros(drift_count)
    windowing_standard_errors = zeros(drift_count)
    intersection_average_costs = zeros(drift_count)
    intersection_standard_errors = zeros(drift_count)
    weighted_average_costs = zeros(drift_count)
    weighted_standard_errors = zeros(drift_count)
    repetition_underage_costs = sample_repetition_item_costs(
        underage_cost_values,
        underage_cost_stream,
    )
    repetition_overage_costs = sample_repetition_item_costs(
        overage_cost_values,
        overage_cost_stream,
    )
    repetition_mixture_weights = sample_repetition_mixture_weights()

    for drift_index in eachindex(drifts)
        drift = drifts[drift_index]
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

            method_grid_results = (
                smoothing_grid_result,
                SAA_grid_result,
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
                SAA_costs,
                repetition_index,
                SAA_grid_result,
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
        (SAA_average_costs[drift_index],
         SAA_standard_errors[drift_index]) =
            summarize_method(SAA_costs)
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
        SAA = (
            average_costs = SAA_average_costs,
            standard_errors = SAA_standard_errors,
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
    yscale = :log10,
    xlabel = "Binomial drift parameter, \$δ\$",
    ylabel = "Expected cost (% difference)",
    topmargin = 10.0pt,
    leftmargin = 6.0pt,
    bottommargin = 6.0pt,
    rightmargin = 3.0pt,
    legend = :topright,
)

fillalpha = 0.1
normalizer = results.smoothing.average_costs

plot!(
    plt,
    drifts,
    results.SAA.average_costs ./ normalizer;
    ribbon = results.SAA.standard_errors ./ normalizer,
    fillalpha = fillalpha,
    color = palette(:tab10)[7],
    linestyle = :dashdot,
    markershape = :pentagon,
    markersize = 4.0,
    markerstrokewidth = 0.0,
    label = "SAA",
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
# (Log-scaled cost ratios, labelled as percentage differences.)
yticks!(
    [0.8, 0.9, 1.0, 1.1, 1.2, 1.4],
    ["−20", "−10", "0", "+10", "+20", "+40"],
)
ylims!((0.79999, 1.40001))
display(plt)
