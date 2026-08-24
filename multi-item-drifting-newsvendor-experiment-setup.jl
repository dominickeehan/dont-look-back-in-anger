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
    # The dimension argument is an item index, so an instance with
    # more items reuses the streams of instances with fewer items.
    seed =
        simulation_seed +
        1_000_000 * global_repetition_index +
        1_000 * dimension +
        stream

    return Xoshiro(seed)
end

function sample_repetition_item_costs(cost_values, stream)
    return map(global_repetition_indices) do global_repetition_index
        rng = experiment_rng(global_repetition_index, stream)
        [rand(rng, cost_values) for _ in 1:number_of_items]
    end
end

# A first mixture weight w defines the two mixture weights [w, 1-w].
function sample_repetition_mixture_weights()
    return map(global_repetition_indices) do global_repetition_index
        rng = experiment_rng(global_repetition_index, mixture_weight_stream)
        first_mixture_weight = Float64(rand(rng, mixture_weight_values))
        [first_mixture_weight, 1.0 - first_mixture_weight]
    end
end

# Innovations are drawn from the unit triangle and scaled by the drift
# parameter so every drift value reuses the same innovation sequence.
const unit_drift_distribution = TriangularDist(-1.0, 1.0, 0.0)

# The explicitly stored item probabilities are capped per item by
# [minimum_purchase_probability, maximum_purchase_probability] and jointly by
# maximum_probability_sum. The mass below that sum is the implicit no-purchase
# category, so the joint cap is one rather than the per-item cap.
const maximum_probability_sum = 1.0

# Euclidean projection onto that bounded sub-simplex.
function project_purchase_probabilities!(purchase_probabilities)
    box_projection = clamp.(
        purchase_probabilities,
        minimum_purchase_probability,
        maximum_purchase_probability,
    )
    if sum(box_projection) <= maximum_probability_sum
        return purchase_probabilities .= box_projection
    end

    # The sum constraint binds. Its Lagrange multiplier is the scalar shift in
    # clamp.(purchase_probabilities .- shift, lower_bound, upper_bound).
    lower_shift = 0.0
    upper_shift = maximum(purchase_probabilities) - minimum_purchase_probability
    for _ in 1:100
        shift = (lower_shift + upper_shift) / 2.0
        projected_sum = sum(
            clamp(probability - shift, minimum_purchase_probability, maximum_purchase_probability)
            for probability in purchase_probabilities
        )
        projected_sum > maximum_probability_sum ? (lower_shift = shift) : (upper_shift = shift)
    end

    return purchase_probabilities .= clamp.(
        purchase_probabilities .- upper_shift,
        minimum_purchase_probability,
        maximum_purchase_probability,
    )
end

# Inverse transform of the sequential conditional-Binomial decomposition of the
# multinomial. This consumes one uniform draw per item regardless of the purchase 
# probabilities, and it is monotone in them, so the same uniforms couple demand 
# across drift values and across item counts.
function sample_multinomial_demand(uniforms, purchase_probabilities)
    demand = Vector{Float64}(undef, number_of_items)
    remaining_consumers = number_of_consumers
    remaining_probability = 1.0

    for item_index in 1:number_of_items
        # The projected probabilities can sum to one, which leaves the
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
# Dirichlet(number_of_items + 1, 1), built from the Gamma representation.
# Item i draws its unit-exponential variate from a stream keyed by i, and 
# the implicit no-purchase category draws one from a stream of its own, so
#
#   probability_i = gamma_i /
#       (no_purchase_gamma + sum(gamma_1, ..., gamma_k))
#
# has the correct Dirichlet law for every item count k while reusing the same
# variates across item counts.
function initial_multinomial_demand_probabilities(global_repetition_index)

    no_purchase_variates = randexp(
        experiment_rng(
            global_repetition_index,
            initial_probability_stream,
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
    return sum(
        instance_underage_costs[item_index] *
        max(demand[item_index] - order[item_index], 0.0) +
        instance_overage_costs[item_index] *
        max(order[item_index] - demand[item_index], 0.0)
        for item_index in 1:number_of_items
    )
end

# Closed form from E[D 1{D <= order}] = consumer_count * probability *
# F_{n-1}(order - 1). See "test-expected-newsvendor-loss.jl".
function expected_newsvendor_cost_with_binomial_demand(
    order,
    binomial_demand_probability,
    consumer_count,
    underage_cost,
    overage_cost,
)
    previous_trial_cdf = cdf(
        Binomial(consumer_count - 1, binomial_demand_probability), order - 1,
    )
    demand_cdf = cdf(
        Binomial(consumer_count, binomial_demand_probability), order,
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
        # number of variates per drift, and the per-item streams are
        # keyed by item index. Thus all drift values and item counts share 
        # multinomial indices, demand uniforms, and innovations.
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
                probabilities = demand_probabilities[multinomial_index]
                probabilities .+= drift .* view(
                    history_innovations, time_index, multinomial_index, :,
                )
                project_purchase_probabilities!(probabilities)
            end
        end

        # Each future sample is a separate one-step drift from the distribution
        # at the end of the observed history.
        for multinomial_index in 1:number_of_multinomials
            future_probabilities[:, multinomial_index, :] .=
                demand_probabilities[multinomial_index]' .+
                drift .* view(future_innovations, :, multinomial_index, :)
            for future_index in 1:number_of_future_samples
                project_purchase_probabilities!(
                    view(future_probabilities, future_index, multinomial_index, :),
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
