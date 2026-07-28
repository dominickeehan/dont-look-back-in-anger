# A single-process preview of the triptych that the HPC array job produces.
# It runs the same experiment as
# generate-train-and-test-multi-item-triptych-drifting-newsvendor-data.jl on
# the same demand model and the same random streams, and it draws the same
# figure as
# process-and-plot-train-and-test-multi-item-triptych-drifting-newsvendor-data.jl,
# so a short local run shows what the full run will look like.
#
# Because the model keys every stream by item index rather than by the number
# of items, an item count of k is the k-item prefix of the ten-item instance,
# and the repetitions requested here are the first repetitions of the HPC run.
# Shrinking either range therefore subsets the full experiment instead of
# replacing it.

using Random, Statistics, StatsBase, Distributions
using ProgressBars


include("experiment-randomness.jl")


# ---------------------------------------------------------------------------
# Preview settings. These are the two knobs worth turning: the full experiment
# is item_counts_to_plot = 1:10 with number_of_repetitions = 1000, which is why
# it runs on a cluster. Runtime is roughly linear in the number of repetitions
# and superlinear in the item count, so 1:4 with 20 repetitions previews in
# minutes. Everything below this block matches the HPC run exactly.
const item_counts_to_plot = 1:4
const number_of_repetitions = 300
# The HPC run evaluates 1000 next-period samples. Lowering this speeds up the
# test-cost evaluation but adds noise to every method equally.
const number_of_future_samples = 1000
# ---------------------------------------------------------------------------


# The optimization routines and the demand model read this at call time, so it
# can change from one item count to the next.
number_of_items::Int = first(item_counts_to_plot)

const number_of_consumers = 1000
const underage_cost_values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
const overage_cost_values = [1.0]
const minimum_purchase_probability = 0.01
const maximum_purchase_probability = 0.99

const first_mode_weight_values = [0.9, 0.95, 0.99]
const number_of_modes = 2

# Starting purchase probabilities are sampled for every mode.
const initial_demand_probabilities = nothing

const drifts = [0.01, 0.05, 0.1]

const history_length = 100
const training_length = 30
# Job j of the array run covers repetitions 10j + 1 to 10(j + 1), so these are
# the repetitions of its first jobs.
const global_repetition_indices = 1:number_of_repetitions

include("weights.jl")
include("multi-item-newsvendor-dual-optimizations.jl")
include("drifting-demand-model.jl")


const methods = ["SAA", "Smoothing", "Intersection", "Weighted"]


function _mark_order_knots!(requested_orders, grid_results)
    for result in grid_results
        order = result[2]
        for item_index in 1:number_of_items
            requested_orders[item_index][floor(Int, order[item_index]) + 1] =
                true
            requested_orders[item_index][ceil(Int, order[item_index]) + 1] =
                true
        end
    end
    return nothing
end


# All methods use the same simulated future distributions. Build one lookup
# from the union of their integer order knots. The expected cost is piecewise
# linear in the order quantity because demand is integer valued, so the
# interpolation below is exact.
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
    inverse_future_sample_count =
        1.0 / size(final_demand_probabilities, 1)
    for item_index in 1:number_of_items
        for order_storage_index in eachindex(requested_orders[item_index])
            requested_orders[item_index][order_storage_index] || continue
            integer_order = order_storage_index - 1
            total_cost = 0.0
            for future_index in axes(final_demand_probabilities, 1)
                for mode_index in 1:number_of_modes
                    total_cost +=
                        mode_weights[mode_index] *
                        expected_newsvendor_cost_with_binomial_demand(
                            integer_order,
                            final_demand_probabilities[
                                future_index,
                                mode_index,
                                item_index,
                            ],
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


function expected_multi_item_cost_from_order_knots(order, expected_costs)
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


# The cost and mode-weight streams do not depend on the number of items, so the
# first k draws of a repetition are its k-item instance.
function sample_repetition_item_costs(cost_values, stream)
    return [
        begin
            rng = experiment_rng(global_repetition_index, stream)
            [rand(rng, cost_values) for _ in 1:number_of_items]
        end
        for global_repetition_index in global_repetition_indices
    ]
end


function sample_repetition_mixture_weights()
    return [
        begin
            rng = experiment_rng(
                global_repetition_index,
                mode_weight_stream,
            )
            first_mode_weight = rand(rng, first_mode_weight_values)
            [first_mode_weight, 1.0 - first_mode_weight]
        end
        for global_repetition_index in global_repetition_indices
    ]
end


LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))

const zero_ambiguity_radius = [0.0]
const smoothing_parameter_grid = [0.0; LogRange(1.0e-4, 1.0e0, 30)]
const radius_ratio_grid = [0.0; LogRange(1.0e-4, 1.0e0, 30)]

epsilon_grid_for_number_of_items(item_count) =
    sqrt(item_count) * number_of_consumers * unique([
        0.0;
        LinRange(1.0e-3, 1.0e-2, 10);
        LinRange(1.0e-2, 1.0e-1, 10);
        LinRange(1.0e-1, 1.0e0, 10)
    ])


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


const saa_weight_vector_table = precompute_weight_vector_table(
    windowing_weights,
    [history_length],
)
const smoothing_weight_vector_table = precompute_weight_vector_table(
    smoothing_weights,
    smoothing_parameter_grid,
)
const intersection_weight_vector_table = precompute_weight_vector_table(
    REMK_intersection_weights,
    radius_ratio_grid,
)
const weighted_W2_weight_vector_table = precompute_weight_vector_table(
    W2_weights,
    radius_ratio_grid,
)


# Select hyperparameters by rolling-origin validation, then refit the winner on
# the full history. The final one-cell grid has the same shape as an ex-post
# grid result, so all methods can share the future-cost lookup above. argmin
# breaks ties in column-major order, which is the order in which the array run
# writes its rows, so both select the same hyperparameters.
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
                ) / training_length
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
function compute_train_and_test_lines(item_count)
    global number_of_items = item_count

    epsilon_grid = epsilon_grid_for_number_of_items(item_count)

    drift_count = length(drifts)
    average_costs = Dict(
        method => zeros(drift_count) for method in methods
    )
    standard_errors = Dict(
        method => zeros(drift_count) for method in methods
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

    for drift_index in eachindex(drifts)
        drift = drifts[drift_index]
        println("Number of items: $item_count; drift: $drift")
        demand_sequences, final_demand_probabilities, _ =
            generate_drift_data(drift, repetition_mixture_weights)

        costs = Dict(
            method => zeros(number_of_repetitions) for method in methods
        )

        Threads.@threads :static for repetition_index in ProgressBar(
            1:number_of_repetitions,
        )
            demand_samples = demand_sequences[repetition_index]
            instance_underage_costs =
                repetition_underage_costs[repetition_index]
            instance_overage_costs =
                repetition_overage_costs[repetition_index]

            grid_results = Dict(
                "SAA" => _train_and_test_grid_result(
                    SO_multi_item_newsvendor_objective_value_and_order,
                    zero_ambiguity_radius,
                    saa_weight_vector_table,
                    demand_samples,
                    instance_underage_costs,
                    instance_overage_costs,
                ),
                "Smoothing" => _train_and_test_grid_result(
                    SO_multi_item_newsvendor_objective_value_and_order,
                    zero_ambiguity_radius,
                    smoothing_weight_vector_table,
                    demand_samples,
                    instance_underage_costs,
                    instance_overage_costs,
                ),
                "Intersection" => _train_and_test_grid_result(
                    REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
                    epsilon_grid,
                    intersection_weight_vector_table,
                    demand_samples,
                    instance_underage_costs,
                    instance_overage_costs,
                ),
                "Weighted" => _train_and_test_grid_result(
                    W2_DRO_multi_item_newsvendor_objective_value_and_order,
                    epsilon_grid,
                    weighted_W2_weight_vector_table,
                    demand_samples,
                    instance_underage_costs,
                    instance_overage_costs,
                ),
            )

            expected_costs = precompute_expected_costs_at_order_knots(
                Tuple(grid_results[method] for method in methods),
                final_demand_probabilities[repetition_index],
                repetition_mixture_weights[repetition_index],
                instance_underage_costs,
                instance_overage_costs,
            )

            for method in methods
                _fill_train_and_test_cost!(
                    costs[method],
                    repetition_index,
                    grid_results[method],
                    expected_costs,
                )
            end
        end

        for method in methods
            (average_costs[method][drift_index],
             standard_errors[method][drift_index]) =
                summarize_method(costs[method])
        end
    end

    return (
        average_costs = average_costs,
        standard_errors = standard_errors,
    )
end


# Collect the same shape the array-run processing script produces, so the
# plotting code below is the plotting code of the full figure.
function compute_train_and_test_results()
    item_counts = collect(item_counts_to_plot)
    results = Dict{Tuple{Int,String},NamedTuple}()
    for item_count in item_counts
        item_count_results = compute_train_and_test_lines(item_count)
        for method in methods
            results[(item_count, method)] = (
                average_costs = item_count_results.average_costs[method],
                standard_errors =
                    item_count_results.standard_errors[method],
            )
        end
    end
    return (
        item_counts = item_counts,
        drifts = drifts,
        methods = methods,
        results = results,
    )
end


processed_results = compute_train_and_test_results()
# Restore the first item count so the experiment-level binding has a
# predictable value after inclusion.
number_of_items = first(item_counts_to_plot)


using Plots, Measures


# The styling below matches
# process-and-plot-train-and-test-multi-item-triptych-drifting-newsvendor-data.jl
# so that a preview and the full figure are directly comparable.
function plot_train_and_test_results(processed_results)
    default()
    a4_width_points = 210.0 / 25.4 * 72.0
    gr(size = (
        a4_width_points - 2.0 * 72.0,
        183 + 6 + 10,
    ) .* sqrt(3))

    fontfamily = "Computer Modern"
    default(
        framestyle = :box,
        grid = true,
        gridalpha = 0.075,
        minorgrid = true,
        minorgridalpha = 0.075,
        minorgridlinestyle = :dash,
        tick_direction = :in,
        xminorticks = 0,
        yminorticks = 0,
        fontfamily = fontfamily,
        guidefont = Plots.font(fontfamily; pointsize = 12),
        legendfont = Plots.font(fontfamily; pointsize = 11),
        tickfont = Plots.font(fontfamily; pointsize = 10),
        titlefont = Plots.font(fontfamily; pointsize = 12),
    )

    styles = Dict(
        "SAA" => (
            color = palette(:tab10)[7],
            linestyle = :dashdot,
            linewidth = 1.0,
            markershape = :pentagon,
            markersize = 4.0,
        ),
        "Smoothing" => (
            color = palette(:tab10)[9],
            linestyle = :dot,
            linewidth = 1.2,
            markershape = :star4,
            markersize = 6.0,
        ),
        "Intersection" => (
            color = palette(:tab10)[1],
            linestyle = :solid,
            linewidth = 1.0,
            markershape = :circle,
            markersize = 4.0,
        ),
        "Weighted" => (
            color = palette(:tab10)[2],
            linestyle = :dash,
            linewidth = 1.0,
            markershape = :diamond,
            markersize = 4.0,
        ),
    )
    marker_outline_color = :black
    marker_outline_width = 1.0

    ytick_values = collect(0.8:0.1:1.4)
    panel_plots = []
    for (drift_index, drift) in enumerate(processed_results.drifts)
        panel_yticks = if drift_index == firstindex(
            processed_results.drifts,
        )
            ytick_values
        else
            (ytick_values, fill("", length(ytick_values)))
        end
        panel = plot(
            xlabel = "Number of items",
            ylabel = drift_index == firstindex(
                processed_results.drifts,
            ) ?
                "Average train-and-test next-period\n" *
                "expected cost (relative to smoothing)" : "",
            title = "\$δ = $drift\$",
            xticks = processed_results.item_counts,
            xlims = (
                0.9999 * first(processed_results.item_counts),
                1.0001 * last(processed_results.item_counts),
            ),
            yticks = panel_yticks,
            ylims = (0.79999, 1.40001),
            legend = drift_index == lastindex(
                processed_results.drifts,
            ) ? :topright : false,
            topmargin = 6.0pt,
            leftmargin = drift_index == firstindex(
                processed_results.drifts,
            ) ? 12.0pt : 1.0pt,
            bottommargin = 12.0pt,
            rightmargin = 0.0pt,
        )

        for method in processed_results.methods
            style = styles[method]
            relative_average_costs = [
                begin
                    method_result = processed_results.results[
                        (item_count, method)
                    ]
                    normalizer = processed_results.results[
                        (item_count, "Smoothing")
                    ].average_costs[drift_index]
                    method_result.average_costs[drift_index] /
                        normalizer
                end
                for item_count in processed_results.item_counts
            ]
            relative_standard_errors = [
                begin
                    method_result = processed_results.results[
                        (item_count, method)
                    ]
                    normalizer = processed_results.results[
                        (item_count, "Smoothing")
                    ].average_costs[drift_index]
                    method_result.standard_errors[drift_index] /
                        normalizer
                end
                for item_count in processed_results.item_counts
            ]
            # Plots uses markerstrokecolor and markerstrokewidth for error
            # bars, so draw them separately with method colours.
            scatter!(
                panel,
                processed_results.item_counts,
                relative_average_costs;
                yerror = relative_standard_errors,
                color = style.color,
                markershape = :none,
                markersize = 3.0,
                markerstrokecolor = style.color,
                markerstrokewidth = marker_outline_width,
                label = false,
            )
            plot!(
                panel,
                processed_results.item_counts,
                relative_average_costs;
                color = style.color,
                linestyle = style.linestyle,
                linewidth = style.linewidth,
                markershape = style.markershape,
                markersize = style.markersize,
                markerstrokecolor = marker_outline_color,
                markerstrokewidth = marker_outline_width,
                label = false,
            )
        end

        if drift_index == lastindex(processed_results.drifts)
            panel_xlims = xlims(panel)
            panel_ylims = ylims(panel)
            for method in processed_results.methods
                style = styles[method]
                scatter!(
                    panel,
                    [-10.0, -11.0],
                    [-10.0, -11.0];
                    color = style.color,
                    alpha = 1.0,
                    linewidth = 1.0,
                    markershape = style.markershape,
                    markersize = 4.0,
                    markerstrokewidth = 0.75,
                    label = "$method",
                )
            end
            xlims!(panel, panel_xlims)
            ylims!(panel, panel_ylims)
        end
        push!(panel_plots, panel)
    end

    return plot(
        panel_plots...;
        layout = (1, length(panel_plots)),
        link = :y,
    )
end


plt = plot_train_and_test_results(processed_results)
display(plt)
