# Demand is multinomial across items: each consumer buys at most one item, so
# item demands are negatively correlated within a period. Each repetition
# chooses one candidate first-mode mixture weight and uses its complement for
# the second mode. Per-mode starting purchase probabilities are sampled unless
# fixed one-item probabilities are supplied below. Per-item underage and
# overage costs are sampled from their configured candidate vectors. The
# no-purchase probability is stored implicitly as one minus the sum of the
# item probabilities. The per-item demand marginals remain Binomial, so the
# expected-cost evaluation below stays exact.

using Random, Statistics, StatsBase, Distributions
using ProgressBars


# These bindings must exist before including the optimization routines.
const number_of_items = 1
const number_of_consumers = 1000
const underage_cost_values = [3.0, 4.0, 5.0, 6.0]
const overage_cost_values = [1.0]
const minimum_purchase_probability = 0.01
const maximum_purchase_probability = 0.99

# A candidate is chosen uniformly at the start of each repetition. A chosen
# value w gives the two mode weights [w, 1-w] for that whole repetition.
const mixture_weights = [0.9, 0.95, 0.99] # For example, [0.9, 0.95, 0.99]
const number_of_modes = 2

# For one item, set this to [p_mode_1, p_mode_2] to fix the two modes'
# starting purchase probabilities. Leave it as nothing to sample them using
# the existing Dirichlet approach independently in every repetition.
const initial_demand_probabilities = nothing
construct_drift_distribution(delta) = TriangularDist(-delta, delta, 0.0)
#const drifts = [1.00e-2, 3.16e-2, 1.00e-1, 3.16e-1, 1.00e0]
const drifts = [5.62e-3, 1.00e-2, 1.79e-2, 3.16e-2, 5.62e-2, 1.00e-1, 1.79e-1, 3.16e-1, 5.62e-1]
#const drifts = [1.00e-2, 1.79e-2, 3.16e-2, 5.62e-2, 1.00e-1, 1.79e-1, 3.16e-1, 5.62e-1, 1.00e0]

include("weights.jl")
include("multi-item-newsvendor-dual-optimizations.jl")

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
    demand_distribution,
    underage_cost,
    overage_cost,
)
    consumer_count, binomial_demand_probability = params(demand_distribution)
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


# All methods use the same simulated future distributions. Build one
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
    inverse_future_sample_count =
        1.0 / size(final_demand_probabilities, 1)
    for item_index in 1:number_of_items
        for order_storage_index in eachindex(requested_orders[item_index])
            if requested_orders[item_index][order_storage_index]
                expected_costs[item_index][order_storage_index] = 0.0
            end
        end

        for future_index in axes(final_demand_probabilities, 1)
            for mode_index in axes(final_demand_probabilities, 2)
                demand_distribution = Binomial(
                    number_of_consumers,
                    final_demand_probabilities[
                        future_index, mode_index, item_index
                    ],
                )
                cost_weight =
                    inverse_future_sample_count * mode_weights[mode_index]
                for order_storage_index in eachindex(
                    requested_orders[item_index],
                )
                    requested_orders[item_index][order_storage_index] || continue
                    expected_costs[item_index][order_storage_index] +=
                        cost_weight *
                        expected_newsvendor_cost_with_binomial_demand(
                            order_storage_index - 1,
                            demand_distribution,
                            instance_underage_costs[item_index],
                            instance_overage_costs[item_index],
                        )
                end
            end
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


function sample_repetition_item_costs(cost_values, seed_offset)
    cost_rng = MersenneTwister(simulation_seed + seed_offset)
    return [
        [
            rand(cost_rng, cost_values)
            for _ in 1:number_of_items
        ]
        for _ in 1:number_of_repetitions
    ]
end


sample_repetition_underage_costs() =
    sample_repetition_item_costs(underage_cost_values, 1)


sample_repetition_overage_costs() =
    sample_repetition_item_costs(overage_cost_values, 2)


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


function validate_drift_configuration()
    for (name, cost_values) in (
        ("underage_cost_values", underage_cost_values),
        ("overage_cost_values", overage_cost_values),
    )
        isempty(cost_values) && error(
            "$name must contain at least one candidate cost.",
        )
        all(cost -> isfinite(cost) && cost > 0.0, cost_values) || error(
            "Every candidate in $name must be finite and positive.",
        )
    end

    isempty(mixture_weights) && error(
        "mixture_weights must contain at least one candidate weight.",
    )
    all(
        weight -> isfinite(weight) && 0.0 <= weight <= 1.0,
        mixture_weights,
    ) || error(
        "Every candidate in mixture_weights must be finite and in [0, 1].",
    )

    isnothing(initial_demand_probabilities) && return nothing
    number_of_items == 1 || error(
        "initial_demand_probabilities is supported only when " *
        "number_of_items == 1.",
    )
    length(initial_demand_probabilities) == number_of_modes || error(
        "initial_demand_probabilities must contain one probability for " *
        "each of the two modes.",
    )
    all(
        probability ->
            isfinite(probability) &&
            minimum_purchase_probability <= probability <=
                maximum_purchase_probability,
        initial_demand_probabilities,
    ) || error(
        "Every initial demand probability must be finite and within the " *
        "configured purchase-probability bounds.",
    )
    return nothing
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


function generate_drift_data(drift)
    validate_drift_configuration()
    Random.seed!(simulation_seed)
    drift_distribution = construct_drift_distribution(drift)

    demand_sequences = Vector{Vector{Vector{Float64}}}(
        undef,
        number_of_repetitions,
    )
    final_demand_probabilities = Vector{Array{Float64,3}}(
        undef,
        number_of_repetitions,
    )
    repetition_mode_weights = Vector{Vector{Float64}}(
        undef,
        number_of_repetitions,
    )

    for repetition_index in 1:number_of_repetitions
        first_mode_weight = Float64(rand(mixture_weights))
        mode_weights = [first_mode_weight, 1.0 - first_mode_weight]
        mode_sampler = Weights(mode_weights)
        demand_probabilities = initial_mode_demand_probabilities()
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
                    mode_probabilities[item_index] +=
                        rand(drift_distribution)
                end
                project_purchase_probabilities!(mode_probabilities)
            end
        end

        for future_index in 1:number_of_future_samples
            for mode_index in 1:number_of_modes
                for item_index in 1:number_of_items
                    future_probabilities[
                        future_index, mode_index, item_index
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
        repetition_mode_weights[repetition_index] = mode_weights
    end
    return (
        demand_sequences,
        final_demand_probabilities,
        repetition_mode_weights,
    )
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


function select_ex_post_costs(costs)
    mean_costs = dropdims(mean(costs; dims = 3); dims = 3)
    ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean_costs))
    return view(
        costs,
        ambiguity_radius_index,
        weight_parameter_index,
        :,
    )
end


# Process every method for a repetition so they share the same history and
# future-demand samples.
function compute_ex_post_lines()
    drift_count = length(drifts)
    method_configurations = (
        smoothing = (
            optimization = SO_multi_item_newsvendor_objective_value_and_order,
            ambiguity_radii = zero_ambiguity_radius,
            weight_vectors = [
                smoothing_weights(history_length, parameter)
                for parameter in smoothing_parameter_grid
            ],
        ),
        saa = (
            optimization = SO_multi_item_newsvendor_objective_value_and_order,
            ambiguity_radii = zero_ambiguity_radius,
            weight_vectors = [
                windowing_weights(history_length, history_length)
            ],
        ),
        windowing = (
            optimization = SO_multi_item_newsvendor_objective_value_and_order,
            ambiguity_radii = zero_ambiguity_radius,
            weight_vectors = [
                windowing_weights(history_length, window_size)
                for window_size in window_size_grid
            ],
        ),
        intersection = (
            optimization =
                REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
            ambiguity_radii = epsilon_grid,
            weight_vectors = [
                REMK_intersection_weights(history_length, parameter)
                for parameter in radius_ratio_grid
            ],
        ),
        weighted = (
            optimization = W2_DRO_multi_item_newsvendor_objective_value_and_order,
            ambiguity_radii = epsilon_grid,
            weight_vectors = [
                W2_weights(history_length, parameter)
                for parameter in radius_ratio_grid
            ],
        ),
    )
    results = map(method_configurations) do _
        (
            average_costs = zeros(drift_count),
            standard_errors = zeros(drift_count),
        )
    end
    repetition_underage_costs = sample_repetition_underage_costs()
    repetition_overage_costs = sample_repetition_overage_costs()

    for drift_index in eachindex(drifts)
        drift = drifts[drift_index]
        println("Binomial drift parameter: $drift")
        demand_sequences, final_demand_probabilities,
            repetition_mode_weights =
            generate_drift_data(drift)

        method_costs = map(method_configurations) do configuration
            zeros(
                length(configuration.ambiguity_radii),
                length(configuration.weight_vectors),
                number_of_repetitions,
            )
        end

        Threads.@threads :static for repetition_index in ProgressBar(
            1:number_of_repetitions,
        )
            demand_samples = demand_sequences[repetition_index]
            instance_underage_costs =
                repetition_underage_costs[repetition_index]
            instance_overage_costs =
                repetition_overage_costs[repetition_index]
            method_grid_results = map(method_configurations) do configuration
                _multi_item_newsvendor_grid(
                    configuration.optimization,
                    configuration.ambiguity_radii,
                    demand_samples,
                    configuration.weight_vectors,
                    instance_underage_costs,
                    instance_overage_costs,
                )
            end
            expected_costs = precompute_expected_costs_at_order_knots(
                method_grid_results,
                final_demand_probabilities[repetition_index],
                repetition_mode_weights[repetition_index],
                instance_underage_costs,
                instance_overage_costs,
            )

            for method_name in keys(method_configurations)
                _fill_ex_post_costs!(
                    getproperty(method_costs, method_name),
                    repetition_index,
                    getproperty(method_grid_results, method_name),
                    expected_costs,
                )
            end
        end

        selected_method_costs = map(select_ex_post_costs, method_costs)
        for method_name in keys(method_configurations)
            selected_costs = getproperty(selected_method_costs, method_name)
            method_results = getproperty(results, method_name)
            method_results.average_costs[drift_index] = mean(selected_costs)
            method_results.standard_errors[drift_index] = sem(selected_costs)
        end
    end

    return results
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
    legendfont = Plots.font(fontfamily; pointsize = 11),
    tickfont = Plots.font(fontfamily; pointsize = 10),
)

plt = plot(
    xscale = :log10,
    xlabel = "Multinomial drift parameter, \$δ\$",
    ylabel = "Ex-post optimal expected\ncost (relative to smoothing)",
    topmargin = 0.0pt,
    leftmargin = 6.0pt,
    bottommargin = 6.0pt,
    rightmargin = 0.0pt,
    legend = :topright,
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

ylims!((0.8, 1.3))
display(plt)
