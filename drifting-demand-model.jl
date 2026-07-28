# The multinomial drifting-newsvendor demand model, shared by the plotting
# script and the HPC generation script so that both produce the same data.
#
# These bindings must exist before this file is included:
#   number_of_items, number_of_consumers, number_of_modes, number_of_repetitions,
#   number_of_future_samples, history_length, global_repetition_indices,
#   minimum_purchase_probability, maximum_purchase_probability,
#   initial_demand_probabilities.
#
# Innovations are drawn from the unit triangle and scaled by the drift
# parameter, so every drift value reuses the same innovation sequence.
#
# Every per-item stream is keyed by the item index rather than by the number of
# items, and every stream draws a number of variates that does not depend on
# the number of items either. A k-item instance is therefore the k-item prefix
# of the same underlying draw as a (k + 1)-item instance: the two share their
# mode sequence, their demand uniforms, their innovations, and their starting
# Gamma variates. Item counts are common random numbers of each other, exactly
# as drift values already are.
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


# The starting probabilities of a mode are Dirichlet(number_of_items + 1, 1),
# built from the Gamma representation rather than from rand(rng, Dirichlet(...)).
# Item i draws its unit-exponential variate from a stream keyed by i, and the
# implicit no-purchase category draws one from a stream of its own, so
#   probability_i = gamma_i / (no_purchase_gamma + sum(gamma_1, ..., gamma_k))
# has the correct Dirichlet law for every item count k while reusing the same
# variates across item counts. Adding an item shrinks the incumbent
# probabilities smoothly instead of resampling them, which is the coupling that
# rand(rng, Dirichlet(k + 1, 1.0)) cannot provide because its dimension, and
# hence its whole draw, changes with k.
function initial_mode_demand_probabilities(global_repetition_index)
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
        number_of_modes,
    )
    item_variates = [
        randexp(
            experiment_rng(
                global_repetition_index,
                initial_probability_stream;
                dimension = item_index,
            ),
            number_of_modes,
        )
        for item_index in 1:number_of_items
    ]

    return [
        begin
            total_variate =
                no_purchase_variates[mode_index] +
                sum(
                    item_variates[item_index][mode_index]
                    for item_index in 1:number_of_items
                )
            project_purchase_probabilities!([
                item_variates[item_index][mode_index] / total_variate
                for item_index in 1:number_of_items
            ])
        end
        for mode_index in 1:number_of_modes
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


# Returns the observed demand histories, the sampled next-period probabilities
# as a (future sample, mode, item) array, and the starting probabilities.
function generate_drift_data(drift, repetition_mode_weights)
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
        # keyed by item index rather than by the number of items. So all drift
        # values and all item counts see the same modes, the same demand
        # uniforms, and the same innovations.
        mode_rng = experiment_rng(global_repetition_index, mode_stream)

        mode_sampler = Weights(repetition_mode_weights[repetition_index])
        modes = [
            sample(mode_rng, 1:number_of_modes, mode_sampler)
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
            number_of_modes,
            number_of_items,
        )
        future_innovations = Array{Float64}(
            undef,
            number_of_future_samples,
            number_of_modes,
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
                number_of_modes,
            )
            future_innovations[:, :, item_index] = rand(
                innovation_rng,
                unit_drift_distribution,
                number_of_future_samples,
                number_of_modes,
            )
        end

        demand_probabilities =
            initial_mode_demand_probabilities(global_repetition_index)
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
            demand_sequence[time_index] =
                sample_multinomial_demand(
                    view(demand_uniforms, time_index, :),
                    demand_probabilities[modes[time_index]],
                )

            time_index == history_length && continue
            for mode_index in 1:number_of_modes
                mode_probabilities = demand_probabilities[mode_index]
                for item_index in eachindex(mode_probabilities)
                    mode_probabilities[item_index] +=
                        drift * history_innovations[
                            time_index,
                            mode_index,
                            item_index,
                        ]
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
                        drift * future_innovations[
                            future_index,
                            mode_index,
                            item_index,
                        ]
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
