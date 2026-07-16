# Train-and-test version of the ex-post multi-item drifting-newsvendor
# experiment. Instead of selecting each method's hyperparameters ex post
# (against the simulated future distributions), every repetition selects them
# by rolling-origin validation: for each of the last `training_length`
# periods, refit on the preceding history and pay the realized newsvendor
# cost at that period. The per-repetition winner is refit on the full history
# and scored against the simulated next-period distributions.
#
# Demand is multinomial across items: each consumer buys at most one item, so
# item demands are negatively correlated within a period. The mixture weights
# over modes are fixed below; each repetition draws its own per-mode starting
# purchase probabilities uniformly at random (with a `probability_floor` on
# every category). The per-item demand marginals remain Binomial, so the
# expected-cost evaluation below stays exact.

using Random, Statistics, StatsBase, Distributions
using ProgressBars


# These bindings must exist before including the optimization routines, which
# copy them into their own typed constants.
const number_of_items = 1
const mixture_weights = [0.9, 0.1]
const number_of_modes = length(mixture_weights)
construct_drift_distribution(delta) = TriangularDist(-delta, delta, 0.0)
const drifts = [1.00e-2, 3.16e-2, 1.00e-1, 3.16e-1, 1.00e0]
#const drifts = [1.00e-2, 1.79e-2, 3.16e-2, 5.62e-2, 1.00e-1, 1.79e-1, 3.16e-1, 5.62e-1, 1.00e0]
const number_of_consumers = 1000
const budget = parse(
    Float64,
    get(ENV, "MULTI_ITEM_NEWSVENDOR_BUDGET", string(number_of_consumers)),
) # Total order quantity (unit procurement costs).
const cu = 4.0 # Per-unit underage cost.
const co = 1.0 # Per-unit overage cost.

include("weights.jl")
include("multi-item-newsvendor-optimizations.jl")

const number_of_repetitions = 1000
const number_of_future_samples = 100
const history_length = 100
const training_length = 30
const simulation_seed = 42
const probability_floor = 0.0 # On every multinomial category.


# For D ~ Binomial(n, p),
#   F_{n-1}(q-1) = F_n(q) - (n-q)/n * P(D=q).
# This replaces the second Binomial CDF in the original expression with an
# allocation-free PDF evaluation while preserving the exact expected cost.
@inline function expected_newsvendor_cost_with_binomial_demand(
    order::Int,
    binomial_demand_probability::Float64,
)::Float64
    0 <= order <= number_of_consumers ||
        throw(ArgumentError("order must lie between zero and number_of_consumers"))
    demand_distribution = Binomial(
        number_of_consumers,
        binomial_demand_probability,
    )
    demand_cdf = cdf(demand_distribution, order)
    previous_trial_cdf = clamp(
        demand_cdf -
        ((number_of_consumers - order) / number_of_consumers) *
        pdf(demand_distribution, order),
        0.0,
        1.0,
    )

    expected_underage_cost = cu * (
        number_of_consumers * binomial_demand_probability *
        (1.0 - previous_trial_cdf) -
        order * (1.0 - demand_cdf)
    )
    expected_overage_cost = co * (
        order * demand_cdf -
        number_of_consumers * binomial_demand_probability * previous_trial_cdf
    )
    return expected_underage_cost + expected_overage_cost
end


function _mark_order_knots!(requested_orders, grid_results)
    for result in grid_results
        order = result[2]
        for item_index in 1:number_of_items
            bounded_order = clamp(
                order[item_index],
                0.0,
                Float64(number_of_consumers),
            )
            requested_orders[item_index][floor(Int, bounded_order) + 1] = true
            requested_orders[item_index][ceil(Int, bounded_order) + 1] = true
        end
    end
    return nothing
end


# Every method is scored against the same simulated future distributions.
# Build one dense lookup from the union of their integer order knots, rather
# than separate tables and repeated future-cost passes.
function precompute_expected_costs_at_order_knots(
    method_grid_results::Tuple,
    final_demand_probabilities::Vector{Vector{Vector{Float64}}},
    mode_weights::Vector{Float64},
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
    expected_costs::Vector{Vector{Float64}},
)::Float64
    total_cost = 0.0
    for item_index in 1:number_of_items
        bounded_order = clamp(
            order[item_index],
            0.0,
            Float64(number_of_consumers),
        )
        lower_order = floor(Int, bounded_order)
        upper_order = ceil(Int, bounded_order)
        lower_cost = expected_costs[item_index][lower_order + 1]
        if lower_order == upper_order
            total_cost += lower_cost
        else
            interpolation_weight = bounded_order - lower_order
            total_cost +=
                (1.0 - interpolation_weight) * lower_cost +
                interpolation_weight *
                expected_costs[item_index][upper_order + 1]
        end
    end
    return total_cost
end


function realized_multi_item_newsvendor_cost(order, demand)::Float64
    total_cost = 0.0
    for item_index in 1:number_of_items
        total_cost +=
            cu * max(demand[item_index] - order[item_index], 0.0) +
            co * max(order[item_index] - demand[item_index], 0.0)
    end
    return total_cost
end


# Uniform sample over the compositions with every category at least
# `probability_floor` (plain uniform Dirichlet when the floor is zero).
function sample_bounded_composition(category_count::Int)
    free_mass = 1.0 - probability_floor * category_count
    free_mass > 0.0 ||
        throw(ArgumentError("probability_floor leaves no mass to distribute"))
    return probability_floor .+
           free_mass .* rand(Dirichlet(category_count, 1.0))
end


# Drifted purchase probabilities must remain a valid multinomial with every
# item probability and the no-purchase remainder at least `probability_floor`.
# Componentwise flooring can push the total past its budget; draining the
# excess uniformly from the unfloored items (refixing any that hit the floor)
# restores feasibility with the smallest uniform adjustment. For a single
# item this reduces to the plain clamp to
# [probability_floor, 1 - probability_floor].
function project_purchase_probabilities!(purchase_probabilities)
    for item_index in eachindex(purchase_probabilities)
        purchase_probabilities[item_index] = max(
            purchase_probabilities[item_index],
            probability_floor,
        )
    end
    excess = sum(purchase_probabilities) - (1.0 - probability_floor)
    while excess > 1.0e-12
        unfloored_count = count(
            probability -> probability > probability_floor,
            purchase_probabilities,
        )
        reduction = excess / unfloored_count
        excess = 0.0
        for item_index in eachindex(purchase_probabilities)
            probability = purchase_probabilities[item_index]
            probability > probability_floor || continue
            reduced_probability = probability - reduction
            if reduced_probability < probability_floor
                excess += probability_floor - reduced_probability
                purchase_probabilities[item_index] = probability_floor
            else
                purchase_probabilities[item_index] = reduced_probability
            end
        end
    end
    # Draining is only float-exact to a few ulps (e.g. 1.3 - 0.3 rounds to
    # 1 + 2^-52), and Binomial rejects any probability past 1. Snap each
    # component back into range.
    for item_index in eachindex(purchase_probabilities)
        purchase_probabilities[item_index] = min(
            purchase_probabilities[item_index],
            1.0 - probability_floor,
        )
    end
    return purchase_probabilities
end


function sample_multinomial_demand(purchase_probabilities)
    category_probabilities = vcat(
        purchase_probabilities,
        max(1.0 - sum(purchase_probabilities), 0.0),
    )
    category_probabilities ./= sum(category_probabilities)
    category_counts = rand(Multinomial(
        number_of_consumers,
        category_probabilities,
    ))
    return Float64.(category_counts[1:number_of_items])
end


function generate_drift_data(drift::Float64)
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
            sample_bounded_composition(number_of_items + 1)[1:number_of_items]
            for _ in 1:number_of_modes
        ]
        demand_sequence = Vector{Vector{Float64}}(undef, history_length)
        future_probabilities = Vector{Vector{Vector{Float64}}}(
            undef,
            number_of_future_samples,
        )

        for time_index in 1:history_length
            mode = sample(1:number_of_modes, mode_sampler)
            demand_sequence[time_index] = sample_multinomial_demand(
                demand_probabilities[mode],
            )

            if time_index < history_length
                for mode_index in 1:number_of_modes
                    mode_probabilities = demand_probabilities[mode_index]
                    for item_index in 1:number_of_items
                        mode_probabilities[item_index] +=
                            rand(drift_distribution)
                    end
                    project_purchase_probabilities!(mode_probabilities)
                end
            else
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
            end
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
const window_size_grid = unique(round.(
    Int,
    LogRange(1, history_length, 30),
))
const smoothing_parameter_grid = [0.0; LogRange(1.0e-4, 1.0e0, 30)]
const radius_ratio_grid = [0.0; LogRange(1.0e-4, 1.0e0, 30)]
const intersection_epsilon_grid = sqrt(number_of_items) * number_of_consumers * unique([
    0.0;
    LinRange(1.0e-3, 1.0e-2, 10);
    LinRange(1.0e-2, 1.0e-1, 10);
    LinRange(1.0e-1, 1.0e0, 10)
])
const intersection_radius_ratio_grid = [
    0.0;
    LogRange(1.0e-4, 1.0e0, 30)
]


# The rolling-origin solves see histories of every length in
# `training_sample_counts`, so each weight parameter needs one weight vector
# per history length (the last row serves the full-history refit).
const training_sample_counts =
    (history_length - training_length):history_length

function precompute_weight_vector_table(compute_weights, parameters)
    table = [
        Vector{Vector{Float64}}(undef, length(parameters))
        for _ in training_sample_counts
    ]
    Threads.@threads for parameter_index in eachindex(parameters)
        for (row_index, sample_count) in enumerate(training_sample_counts)
            table[row_index][parameter_index] = compute_weights(
                sample_count,
                parameters[parameter_index],
            )
        end
    end
    return table
end


# The result and plot ordering below is positional, so this order must stay
# smoothing, windowing, intersection, weighted.
const train_and_test_method_specs = (
    (
        name = "Smoothing",
        objective_value_and_order =
            SO_multi_item_newsvendor_objective_value_and_order,
        ambiguity_radii = zero_ambiguity_radius,
        weight_parameters = smoothing_parameter_grid,
        compute_weights = smoothing_weights,
    ),
    (
        name = "Windowing",
        objective_value_and_order =
            SO_multi_item_newsvendor_objective_value_and_order,
        ambiguity_radii = zero_ambiguity_radius,
        weight_parameters = window_size_grid,
        compute_weights = windowing_weights,
    ),
    (
        name = "Intersection",
        objective_value_and_order =
            REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order,
        ambiguity_radii = intersection_epsilon_grid,
        weight_parameters = intersection_radius_ratio_grid,
        compute_weights = REMK_intersection_weights,
    ),
    (
        name = "Weighted",
        objective_value_and_order =
            W2_DRO_multi_item_newsvendor_objective_value_and_order,
        ambiguity_radii = epsilon_grid,
        weight_parameters = radius_ratio_grid,
        compute_weights = W2_weights,
    ),
)


function _train_and_test_selection(method_spec, weight_table, demand_sequence)
    first_sample_count = history_length - training_length
    training_costs = zeros(
        length(method_spec.ambiguity_radii),
        length(method_spec.weight_parameters),
    )

    for time_index in (first_sample_count + 1):history_length
        sample_count = time_index - 1
        weight_vectors = weight_table[sample_count - first_sample_count + 1]
        grid_results = _multi_item_newsvendor_grid(
            method_spec.objective_value_and_order,
            method_spec.ambiguity_radii,
            demand_sequence[1:sample_count],
            weight_vectors,
            0,
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
                )
        end
    end

    ambiguity_radius_index, weight_parameter_index =
        Tuple(argmin(training_costs))
    final_grid = _multi_item_newsvendor_grid(
        method_spec.objective_value_and_order,
        method_spec.ambiguity_radii[
            ambiguity_radius_index:ambiguity_radius_index,
        ],
        demand_sequence,
        weight_table[end][weight_parameter_index:weight_parameter_index],
        0,
    )
    _, order, doubling_count = final_grid[1, 1]

    return (
        order = order,
        doubling_count = doubling_count,
        ambiguity_radius =
            method_spec.ambiguity_radii[ambiguity_radius_index],
        weight_parameter =
            method_spec.weight_parameters[weight_parameter_index],
    )
end


function _display_and_reset_solver_statistics!()
    solver_statistics = multi_item_solver_statistics_summary()
    if solver_statistics.touching_solutions +
       solver_statistics.additive_radius_repairs +
       solver_statistics.zero_multiplier_solutions +
       solver_statistics.single_ball_solutions +
       solver_statistics.dual_solver_solutions +
       solver_statistics.conic_solutions +
       solver_statistics.numeric_retry_solves +
       solver_statistics.pair_certificate_solutions +
       solver_statistics.additive_candidate_certificate_solutions +
       solver_statistics.additive_geometry_socp_solves > 0
        println(solver_statistics)
    end
    multi_item_reset_solver_statistics!()
    return nothing
end


function _summarize_train_and_test_method!(
    average_costs,
    standard_deviations,
    drift_index,
    method_name,
    costs,
    selected_ambiguity_radii,
    selected_weight_parameters,
    selected_doubling_counts,
)
    digits = 4
    average_cost = round(mean(costs); digits = digits)
    standard_deviation = round(sem(costs); digits = digits)

    println(method_name)
    print("Train-and-test average next-period expected cost: ")
    print("$average_cost ± $standard_deviation, ")
    print("Mean selected ambiguity radius: ")
    print("$(round(mean(selected_ambiguity_radii); digits = digits)), ")
    print("Mean selected weight parameter: ")
    print("$(round(mean(selected_weight_parameters); digits = digits)), ")
    println(
        "Mean doubling count: " *
        "$(round(mean(selected_doubling_counts); digits = digits))",
    )

    average_costs[drift_index] = average_cost
    standard_deviations[drift_index] = standard_deviation
    return nothing
end


function compute_train_and_test_lines()
    weight_tables = map(
        method_spec -> precompute_weight_vector_table(
            method_spec.compute_weights,
            method_spec.weight_parameters,
        ),
        train_and_test_method_specs,
    )

    drift_count = length(drifts)
    method_count = length(train_and_test_method_specs)
    average_costs = [zeros(drift_count) for _ in 1:method_count]
    standard_deviations = [zeros(drift_count) for _ in 1:method_count]

    for drift_index in eachindex(drifts)
        drift = drifts[drift_index]
        println("Binomial drift parameter: $drift")
        demand_sequences, final_demand_probabilities =
            generate_drift_data(drift)

        costs = [zeros(number_of_repetitions) for _ in 1:method_count]
        selected_ambiguity_radii =
            [zeros(number_of_repetitions) for _ in 1:method_count]
        selected_weight_parameters =
            [zeros(number_of_repetitions) for _ in 1:method_count]
        selected_doubling_counts =
            [zeros(Int, number_of_repetitions) for _ in 1:method_count]

        multi_item_reset_solver_statistics!()
        Threads.@threads :static for repetition_index in ProgressBar(
            1:number_of_repetitions,
        )
            demand_sequence = demand_sequences[repetition_index]
            selections = map(
                (method_spec, weight_table) -> _train_and_test_selection(
                    method_spec,
                    weight_table,
                    demand_sequence,
                ),
                train_and_test_method_specs,
                weight_tables,
            )

            # Score every method's selected order through one shared dense
            # expected-cost lookup, exactly as in the ex-post experiment.
            selected_grid_results = map(selections) do selection
                grid = Matrix{Tuple{Float64,Vector{Float64},Int}}(
                    undef, 1, 1,
                )
                grid[1, 1] = (0.0, selection.order, 0)
                grid
            end
            expected_costs = precompute_expected_costs_at_order_knots(
                selected_grid_results,
                final_demand_probabilities[repetition_index],
                mixture_weights,
            )

            for method_index in 1:method_count
                selection = selections[method_index]
                costs[method_index][repetition_index] =
                    expected_multi_item_cost_from_order_knots(
                        selection.order,
                        expected_costs,
                    )
                selected_ambiguity_radii[method_index][repetition_index] =
                    selection.ambiguity_radius
                selected_weight_parameters[method_index][repetition_index] =
                    selection.weight_parameter
                selected_doubling_counts[method_index][repetition_index] =
                    selection.doubling_count
            end
        end

        _display_and_reset_solver_statistics!()
        for method_index in 1:method_count
            _summarize_train_and_test_method!(
                average_costs[method_index],
                standard_deviations[method_index],
                drift_index,
                train_and_test_method_specs[method_index].name,
                costs[method_index],
                selected_ambiguity_radii[method_index],
                selected_weight_parameters[method_index],
                selected_doubling_counts[method_index],
            )
        end
    end

    return (
        smoothing = (
            average_costs = average_costs[1],
            standard_deviations = standard_deviations[1],
        ),
        windowing = (
            average_costs = average_costs[2],
            standard_deviations = standard_deviations[2],
        ),
        intersection = (
            average_costs = average_costs[3],
            standard_deviations = standard_deviations[3],
        ),
        weighted = (
            average_costs = average_costs[4],
            standard_deviations = standard_deviations[4],
        ),
    )
end


if Threads.nthreads() == 1
    @warn(
        "This experiment is running with one Julia thread. " *
        "Restart with `julia --threads=auto " *
        "plot-train-and-test-multi-item-drifting-newsvendor.jl` " *
        "for parallel repetitions.",
    )
end

results = compute_train_and_test_lines()


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
    ylabel = "Train-and-test next-period expected\ncost (relative to smoothing)",
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
    results.windowing.average_costs ./ normalizer;
    ribbon = results.windowing.standard_deviations ./ normalizer,
    fillalpha = fillalpha,
    color = palette(:tab10)[7],
    linestyle = :dashdot,
    markershape = :pentagon,
    markersize = 4.0,
    markerstrokewidth = 0.0,
    label = "Windowing (\$ε=0\$)",
)
plot!(
    plt,
    drifts,
    results.smoothing.average_costs ./ normalizer;
    ribbon = results.smoothing.standard_deviations ./ normalizer,
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
    ribbon = results.intersection.standard_deviations ./ normalizer,
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
    ribbon = results.weighted.standard_deviations ./ normalizer,
    fillalpha = fillalpha,
    color = palette(:tab10)[2],
    linestyle = :dash,
    markershape = :diamond,
    markersize = 4.0,
    markerstrokewidth = 0.0,
    label = "Weighted",
)


ylims!((0.8,1.2))

display(plt)

#savefig(plt, "figures/train-and-test-multi-item-drifting-newsvendor.pdf")
