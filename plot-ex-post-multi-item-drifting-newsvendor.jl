display(Threads.nthreads())

# Demand is multinomial across items: each consumer buys at most one item, so
# item demands are negatively correlated within a period. The mixture weights
# over modes are fixed below; each repetition draws its own per-mode starting
# purchase probabilities and per-item underage and overage costs uniformly at
# random. The no-purchase probability is stored implicitly as one minus the
# sum of the item probabilities. The per-item demand marginals remain Binomial,
# so the expected-cost evaluation below stays exact.

using Random, Statistics, StatsBase, Distributions
using ProgressBars


# These bindings must exist before including the optimization routines, which
# copy them into their own typed constants.
const number_of_items = 1
const number_of_consumers = 1000
const underage_cost_values = [3.0, 4.0, 5.0, 6.0]
const overage_cost = 1.0
const minimum_purchase_probability = 0.01
const maximum_purchase_probability = 0.99

const mixture_weights = [0.9]
const number_of_modes = length(mixture_weights)
construct_drift_distribution(delta) = TriangularDist(-delta, delta, 0.0)
const drifts = [1.00e-2, 3.16e-2, 1.00e-1, 3.16e-1, 1.00e0]
#const drifts = [1.00e-2, 1.79e-2, 3.16e-2, 5.62e-2, 1.00e-1, 1.79e-1, 3.16e-1, 5.62e-1, 1.00e0]

include("weights.jl")
include("multi-item-newsvendor-optimizations.jl")

const number_of_repetitions = 1000
const number_of_future_samples = 100
const history_length = 100
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


# All three methods use the same simulated future distributions. Build one
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


sample_repetition_overage_costs() =
    [fill(overage_cost, number_of_items) for _ in 1:number_of_repetitions]


# Euclidean projection onto the bounded sub-simplex for the explicitly stored
# item probabilities. Probability mass below one belongs to the implicit
# no-purchase category.
function project_purchase_probabilities!(purchase_probabilities)
    maximum_probability_sum = 1.0
    length(purchase_probabilities) * minimum_purchase_probability <=
        maximum_probability_sum || error(
            "The purchase-probability bounds define an empty sub-simplex.",
        )

    box_projection = clamp.(
        purchase_probabilities,
        minimum_purchase_probability,
        maximum_purchase_probability,
    )
    if sum(box_projection) <= maximum_probability_sum
        purchase_probabilities .= box_projection
        return purchase_probabilities
    end

    # The sum constraint binds. Its Lagrange multiplier is the scalar shift in
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


function generate_drift_data(drift)
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
    mode_sampler = Weights(mixture_weights)

    for repetition_index in 1:number_of_repetitions
        demand_probabilities = [
            project_purchase_probabilities!(
                rand(Dirichlet(number_of_items + 1, 1.0))[1:number_of_items],
            ) for _ in 1:number_of_modes
        ]
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
                sample_multinomial_demand(demand_probabilities[mode])

            time_index == history_length && continue
            for mode_index in 1:number_of_modes
                mode_probabilities = demand_probabilities[mode_index]
                for item_index in eachindex(mode_probabilities)
                    mode_probabilities[item_index] +=
                        rand(drift_distribution)
                end
                project_purchase_probabilities!(mode_probabilities)
            end
        end

        for future_index in 1:number_of_future_samples
            future_probabilities[future_index] = [
                project_purchase_probabilities!([
                    demand_probabilities[mode_index][item_index] +
                    rand(drift_distribution)
                    for item_index in 1:number_of_items
                ])
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


function _fill_ex_post_costs!(
    costs,
    repetition_index,
    grid_results,
    expected_costs,
)
    for weight_parameter_index in axes(grid_results, 2),
        ambiguity_radius_index in axes(grid_results, 1)
        _, order =
            grid_results[ambiguity_radius_index, weight_parameter_index]
        costs[
            ambiguity_radius_index,
            weight_parameter_index,
            repetition_index,
        ] = expected_multi_item_cost_from_order_knots(
            order,
            expected_costs,
        )
    end
    return nothing
end


function summarize_method(costs)
    mean_costs = dropdims(mean(costs; dims = 3); dims = 3)
    ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean_costs))
    minimal_costs = view(
        costs,
        ambiguity_radius_index,
        weight_parameter_index,
        :,
    )
    return mean(minimal_costs), sem(minimal_costs)
end


# Process every method for a repetition so they share the same history and
# future-demand samples.
function compute_ex_post_lines()
    smoothing_weight_vectors = [
        smoothing_weights(history_length, parameter)
        for parameter in smoothing_parameter_grid
    ]
    intersection_weight_vectors = [
        REMK_intersection_weights(history_length, parameter)
        for parameter in radius_ratio_grid
    ]
    weighted_W2_weight_vectors = [
        W2_weights(history_length, parameter)
        for parameter in radius_ratio_grid
    ]

    drift_count = length(drifts)
    smoothing_average_costs = zeros(drift_count)
    smoothing_standard_errors = zeros(drift_count)
    intersection_average_costs = zeros(drift_count)
    intersection_standard_errors = zeros(drift_count)
    weighted_average_costs = zeros(drift_count)
    weighted_standard_errors = zeros(drift_count)
    repetition_underage_costs = sample_repetition_underage_costs()
    repetition_overage_costs = sample_repetition_overage_costs()

    for drift_index in eachindex(drifts)
        drift = drifts[drift_index]
        println("Binomial drift parameter: $drift")
        demand_sequences, final_demand_probabilities =
            generate_drift_data(drift)

        smoothing_costs = zeros(
            length(zero_ambiguity_radius),
            length(smoothing_parameter_grid),
            number_of_repetitions,
        )
        intersection_costs = zeros(
            length(epsilon_grid),
            length(radius_ratio_grid),
            number_of_repetitions,
        )
        weighted_costs = zeros(
            length(epsilon_grid),
            length(radius_ratio_grid),
            number_of_repetitions,
        )

        Threads.@threads :static for repetition_index in ProgressBar(
            1:number_of_repetitions,
        )
            demand_samples = demand_sequences[repetition_index]
            instance_underage_costs =
                repetition_underage_costs[repetition_index]
            instance_overage_costs =
                repetition_overage_costs[repetition_index]
            smoothing_grid_results = _multi_item_newsvendor_grid(
                SO_multi_item_newsvendor_objective_value_and_order,
                zero_ambiguity_radius,
                demand_samples,
                smoothing_weight_vectors,
                instance_underage_costs,
                instance_overage_costs,
            )
            intersection_grid_results = _multi_item_newsvendor_grid(
                REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
                epsilon_grid,
                demand_samples,
                intersection_weight_vectors,
                instance_underage_costs,
                instance_overage_costs,
            )
            weighted_grid_results = _multi_item_newsvendor_grid(
                W2_DRO_multi_item_newsvendor_objective_value_and_order,
                epsilon_grid,
                demand_samples,
                weighted_W2_weight_vectors,
                instance_underage_costs,
                instance_overage_costs,
            )

            method_grid_results = (
                smoothing_grid_results,
                intersection_grid_results,
                weighted_grid_results,
            )
            expected_costs = precompute_expected_costs_at_order_knots(
                method_grid_results,
                final_demand_probabilities[repetition_index],
                mixture_weights,
                instance_underage_costs,
                instance_overage_costs,
            )

            _fill_ex_post_costs!(
                smoothing_costs,
                repetition_index,
                smoothing_grid_results,
                expected_costs,
            )
            _fill_ex_post_costs!(
                intersection_costs,
                repetition_index,
                intersection_grid_results,
                expected_costs,
            )
            _fill_ex_post_costs!(
                weighted_costs,
                repetition_index,
                weighted_grid_results,
                expected_costs,
            )
        end

        (smoothing_average_costs[drift_index],
         smoothing_standard_errors[drift_index]) =
            summarize_method(smoothing_costs)
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
results = compute_ex_post_lines()

using Plots, Measures

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
    legendfont = Plots.font(fontfamily; pointsize = 3),
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
    label = "Smoothing (\$ε=0\$)",
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

ylims!((0.8, 1.2))
display(plt)

5
