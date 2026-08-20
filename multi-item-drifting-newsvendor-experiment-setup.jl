# Shared setup for the multi-item drifting-newsvendor experiments.
#
# The plotting and HPC generation scripts define the experiment configuration
# before including this file. The following bindings must exist:
#   number_of_items, number_of_consumers, number_of_multinomials,
#   number_of_repetitions, number_of_future_samples, history_length,
#   training_length, global_repetition_indices,
#   minimum_purchase_probability, maximum_purchase_probability,
#   initial_demand_probabilities, mixture_weight_values.
#
# The including script must also load Random, Distributions, and StatsBase.
# The rolling-origin helpers expect _multi_item_newsvendor_grid to have been
# supplied by the included optimization implementation.

const simulation_seed = 42

const underage_cost_stream = 1
const overage_cost_stream = 2
const mixture_weight_stream = 3
const demand_stream = 4
const initial_probability_stream = 5
const multinomial_stream = 6
const innovation_stream = 7


function experiment_rng(
    global_repetition_index,
    stream;
    dimension = 0,
)
    # Keep each repetition, random quantity, and item on a deterministic stream
    # that does not depend on surrounding loop order. The dimension argument is
    # an item index, never a number of items, so an instance with more items
    # reuses every stream of the instances with fewer items.
    seed =
        simulation_seed +
        1_000_000 * global_repetition_index +
        1_000 * dimension +
        stream
    return Xoshiro(seed)
end


function sample_repetition_item_costs(cost_values, stream)
    return [
        begin
            rng = experiment_rng(global_repetition_index, stream)
            [rand(rng, cost_values) for _ in 1:number_of_items]
        end
        for global_repetition_index in global_repetition_indices
    ]
end


# A candidate first mixture weight w produces the two mixture weights [w, 1-w].
function sample_repetition_mixture_weights()
    return [
        begin
            rng = experiment_rng(
                global_repetition_index,
                mixture_weight_stream,
            )
            first_mixture_weight =
                Float64(rand(rng, mixture_weight_values))
            [first_mixture_weight, 1.0 - first_mixture_weight]
        end
        for global_repetition_index in global_repetition_indices
    ]
end


# Innovations are drawn from the unit triangle and scaled by the drift
# parameter, so every drift value reuses the same innovation sequence.
const unit_drift_distribution = TriangularDist(-1.0, 1.0, 0.0)


# Euclidean projection onto the bounded sub-simplex for the explicitly stored
# item probabilities. Probability mass below one belongs to the implicit
# no-purchase category.
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


# Inverse transform of the sequential conditional-Binomial decomposition of the
# multinomial. Unlike rand(rng, Multinomial(...)), this consumes exactly one
# uniform per item whatever the purchase probabilities are, and it is monotone
# in them, so the same uniforms couple demand across drift values and across
# item counts.
function sample_multinomial_demand(uniforms, purchase_probabilities)
    demand = Vector{Float64}(undef, number_of_items)
    remaining_consumers = number_of_consumers
    remaining_probability = 1.0
    for item_index in 1:number_of_items
        # The projected probabilities can sum to exactly one, which leaves the
        # last conditional probability a rounding error above one.
        conditional_probability = remaining_probability > 0.0 ?
            clamp(
                purchase_probabilities[item_index] / remaining_probability,
                0.0,
                1.0,
            ) : 0.0
        count = quantile(
            Binomial(remaining_consumers, conditional_probability),
            uniforms[item_index],
        )
        demand[item_index] = Float64(count)
        remaining_consumers -= count
        remaining_probability -= purchase_probabilities[item_index]
    end
    return demand
end


# The starting probabilities of a multinomial are
# Dirichlet(number_of_items + 1, 1), built from the Gamma representation rather
# than from rand(rng, Dirichlet(...)). Item i draws its unit-exponential variate
# from a stream keyed by i, and the implicit no-purchase category draws one from
# a stream of its own, so
#
#   probability_i = gamma_i /
#       (no_purchase_gamma + sum(gamma_1, ..., gamma_k))
#
# has the correct Dirichlet law for every item count k while reusing the same
# variates across item counts.
function initial_multinomial_demand_probabilities(global_repetition_index)
    if !isnothing(initial_demand_probabilities)
        return [
            [Float64(probability)]
            for probability in initial_demand_probabilities
        ]
    end

    no_purchase_variates = randexp(
        experiment_rng(
            global_repetition_index,
            initial_probability_stream;
            dimension = 0,
        ),
        number_of_multinomials,
    )
    item_variates = [
        randexp(
            experiment_rng(
                global_repetition_index,
                initial_probability_stream;
                dimension = item_index,
            ),
            number_of_multinomials,
        )
        for item_index in 1:number_of_items
    ]

    return [
        begin
            total_variate =
                no_purchase_variates[multinomial_index] +
                sum(
                    item_variates[item_index][multinomial_index]
                    for item_index in 1:number_of_items
                )
            project_purchase_probabilities!([
                item_variates[item_index][multinomial_index] / total_variate
                for item_index in 1:number_of_items
            ])
        end
        for multinomial_index in 1:number_of_multinomials
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


# For D ~ Binomial(n, p), the expected newsvendor cost is piecewise linear in
# the order because D is integer valued. The formula is therefore exact for
# both integer and fractional orders. At integer orders, recover the previous
# trial CDF from the demand PDF to avoid constructing a second distribution.
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
    previous_trial_cdf = if isinteger(order)
        clamp(
            demand_cdf -
            ((consumer_count - order) / consumer_count) *
            pdf(demand_distribution, order),
            0.0,
            1.0,
        )
    else
        cdf(
            Binomial(
                consumer_count - 1,
                binomial_demand_probability,
            ),
            order - 1,
        )
    end

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


function expected_multi_item_newsvendor_cost(
    order,
    future_demand_probabilities,
    mixture_weights,
    instance_underage_costs,
    instance_overage_costs,
)
    total_cost = 0.0
    inverse_future_sample_count =
        1.0 / size(future_demand_probabilities, 1)
    for future_index in axes(future_demand_probabilities, 1)
        for multinomial_index in axes(future_demand_probabilities, 2)
            cost_weight =
                inverse_future_sample_count *
                mixture_weights[multinomial_index]
            for item_index in 1:number_of_items
                total_cost +=
                    cost_weight *
                    expected_newsvendor_cost_with_binomial_demand(
                        order[item_index],
                        future_demand_probabilities[
                            future_index,
                            multinomial_index,
                            item_index,
                        ],
                        number_of_consumers,
                        instance_underage_costs[item_index],
                        instance_overage_costs[item_index],
                    )
            end
        end
    end
    return total_cost
end


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


# Build one expected-cost lookup from the union of the methods' integer order
# knots. Fractional expected costs are recovered by linear interpolation.
function precompute_expected_costs_at_order_knots(
    method_grid_results,
    final_demand_probabilities,
    mixture_weights,
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
                for multinomial_index in 1:number_of_multinomials
                    total_cost +=
                        mixture_weights[multinomial_index] *
                        expected_newsvendor_cost_with_binomial_demand(
                            integer_order,
                            final_demand_probabilities[
                                future_index,
                                multinomial_index,
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


# Returns the observed demand histories, the sampled next-period probabilities
# as a (future sample, multinomial, item) array, and the starting probabilities.
function generate_drift_data(drift, repetition_mixture_weights)
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
        global_repetition_index =
            global_repetition_indices[repetition_index]
        # Every random quantity gets its own stream, every stream draws the same
        # number of variates whatever the drift is, and the per-item streams are
        # keyed by item index rather than by the number of items. Thus all drift
        # values and item counts see the same multinomial indices, demand
        # uniforms, and innovations.
        multinomial_rng = experiment_rng(
            global_repetition_index,
            multinomial_stream,
        )

        multinomial_sampler =
            Weights(repetition_mixture_weights[repetition_index])
        multinomial_indices = [
            sample(
                multinomial_rng,
                1:number_of_multinomials,
                multinomial_sampler,
            )
            for _ in 1:history_length
        ]

        demand_uniforms = Matrix{Float64}(
            undef,
            history_length,
            number_of_items,
        )
        history_innovations = Array{Float64}(
            undef,
            history_length - 1,
            number_of_multinomials,
            number_of_items,
        )
        future_innovations = Array{Float64}(
            undef,
            number_of_future_samples,
            number_of_multinomials,
            number_of_items,
        )
        for item_index in 1:number_of_items
            demand_rng = experiment_rng(
                global_repetition_index,
                demand_stream;
                dimension = item_index,
            )
            demand_uniforms[:, item_index] =
                rand(demand_rng, history_length)

            innovation_rng = experiment_rng(
                global_repetition_index,
                innovation_stream;
                dimension = item_index,
            )
            history_innovations[:, :, item_index] = rand(
                innovation_rng,
                unit_drift_distribution,
                history_length - 1,
                number_of_multinomials,
            )
            future_innovations[:, :, item_index] = rand(
                innovation_rng,
                unit_drift_distribution,
                number_of_future_samples,
                number_of_multinomials,
            )
        end

        demand_probabilities =
            initial_multinomial_demand_probabilities(
                global_repetition_index,
            )
        starting_demand_probabilities[repetition_index] =
            deepcopy(demand_probabilities)
        demand_sequence = Vector{Vector{Float64}}(
            undef,
            history_length,
        )
        future_probabilities = Array{Float64}(
            undef,
            number_of_future_samples,
            number_of_multinomials,
            number_of_items,
        )

        for time_index in 1:history_length
            demand_sequence[time_index] =
                sample_multinomial_demand(
                    view(demand_uniforms, time_index, :),
                    demand_probabilities[
                        multinomial_indices[time_index]
                    ],
                )

            time_index == history_length && continue
            for multinomial_index in 1:number_of_multinomials
                multinomial_probabilities =
                    demand_probabilities[multinomial_index]
                for item_index in eachindex(multinomial_probabilities)
                    multinomial_probabilities[item_index] +=
                        drift * history_innovations[
                            time_index,
                            multinomial_index,
                            item_index,
                        ]
                end
                project_purchase_probabilities!(
                    multinomial_probabilities,
                )
            end
        end

        # Each future sample is a separate one-step drift from the distribution
        # at the end of the observed history.
        for future_index in 1:number_of_future_samples
            for multinomial_index in 1:number_of_multinomials
                for item_index in 1:number_of_items
                    future_probabilities[
                        future_index,
                        multinomial_index,
                        item_index,
                    ] =
                        demand_probabilities[
                            multinomial_index
                        ][item_index] +
                        drift * future_innovations[
                            future_index,
                            multinomial_index,
                            item_index,
                        ]
                end
                project_purchase_probabilities!(
                    view(
                        future_probabilities,
                        future_index,
                        multinomial_index,
                        :,
                    ),
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


function precompute_weight_vector_table(compute_weights, parameters)
    first_sample_count = history_length - training_length
    sample_counts = collect(first_sample_count:history_length)
    weight_vector_table = Vector{Vector{Vector{Float64}}}(
        undef,
        length(sample_counts),
    )

    Threads.@threads for sample_count_index in eachindex(sample_counts) # Since .pbs calls for 1 thread, this is allowed on HPC.
        sample_count = sample_counts[sample_count_index]
        weight_vector_table[sample_count_index] = [
            compute_weights(sample_count, parameter)
            for parameter in parameters
        ]
    end

    return weight_vector_table
end


# Evaluate every hyperparameter combination by rolling-origin validation.
function rolling_origin_training_costs(
    objective_value_and_order,
    ambiguity_radii,
    weight_vector_table,
    demand_sequence,
    instance_underage_costs,
    instance_overage_costs,
)
    first_sample_count = history_length - training_length
    average_training_costs = zeros(
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
            average_training_costs[
                ambiguity_radius_index,
                weight_parameter_index,
            ] += realized_multi_item_newsvendor_cost(
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

    return average_training_costs
end


# Select hyperparameters by rolling-origin validation, then refit the winner on
# the full history.
function train_and_test_grid_result(
    objective_value_and_order,
    ambiguity_radii,
    weight_vector_table,
    demand_sequence,
    instance_underage_costs,
    instance_overage_costs,
)
    training_costs = rolling_origin_training_costs(
        objective_value_and_order,
        ambiguity_radii,
        weight_vector_table,
        demand_sequence,
        instance_underage_costs,
        instance_overage_costs,
    )
    ambiguity_radius_index, weight_parameter_index =
        Tuple(argmin(training_costs))

    return _multi_item_newsvendor_grid(
        objective_value_and_order,
        ambiguity_radii[
            ambiguity_radius_index:ambiguity_radius_index
        ],
        demand_sequence,
        weight_vector_table[end][
            weight_parameter_index:weight_parameter_index
        ],
        instance_underage_costs,
        instance_overage_costs,
    )
end
