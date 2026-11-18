# Functions shared by both drifting newsvendor generate scripts.


# Random number streams for each repetition and item.
underage_cost_stream = 1
overage_cost_stream = 2
mixture_weight_stream = 3
demand_stream = 4
initial_probability_stream = 5
multinomial_stream = 6
innovation_stream = 7
experiment_rng(repetition, stream; item = 0) = Xoshiro(42 + 1_000_000*repetition + 1_000*item + stream)


# Costs of each item.
function sample_item_costs(repetition, cost_values, stream)

        rng = experiment_rng(repetition, stream)

        return [rand(rng, cost_values) for _ in 1:number_of_items]
end

# Mixture weights of the two multinomials.
function sample_mixture_weights(repetition)

        first_mixture_weight = rand(experiment_rng(repetition, mixture_weight_stream), mixture_weight_values)

        return [first_mixture_weight, 1 - first_mixture_weight]
end


# Euclidean projection onto bounded purchase probabilities that sum to at most one.
function project_purchase_probabilities!(purchase_probabilities)

        projection(shift) = clamp.(purchase_probabilities .- shift, minimum_purchase_probability, maximum_purchase_probability)

        if sum(projection(0)) <= 1; return purchase_probabilities .= projection(0); end

        # Otherwise bisection finds the shift before clamping that makes the sum one.
        lower_shift = 0.0
        upper_shift = maximum(purchase_probabilities) - minimum_purchase_probability
        for _ in 1:100
                shift = (lower_shift + upper_shift)/2
                if sum(projection(shift)) > 1; lower_shift = shift; else upper_shift = shift; end

        end

        return purchase_probabilities .= projection(upper_shift)
end

# Initial purchase probabilities of each multinomial sampled uniformly and then projected.
function sample_initial_purchase_probabilities(repetition)

        no_purchase_exponentials = randexp(experiment_rng(repetition, initial_probability_stream), number_of_multinomials)
        item_exponentials = [randexp(experiment_rng(repetition, initial_probability_stream, item = item), number_of_multinomials) for item in 1:number_of_items]

        return [project_purchase_probabilities!([item_exponentials[item][multinomial] for item in 1:number_of_items] /
                                                (no_purchase_exponentials[multinomial] + sum(item_exponentials[item][multinomial] for item in 1:number_of_items)))
                for multinomial in 1:number_of_multinomials]
end

# Multinomial demand sampled one item at a time by inverting binomial CDFs at the uniforms.
function sample_multinomial_demand(uniforms, purchase_probabilities)

        demand = zeros(number_of_items)
        remaining_consumers = number_of_consumers
        remaining_probability = 1.0

        for item in 1:number_of_items

                # Conditional purchase probability of this item clamped for rounding errors.
                conditional_probability = remaining_probability > 0 ? clamp(purchase_probabilities[item]/remaining_probability, 0, 1) : 0.0

                item_demand = quantile(Binomial(remaining_consumers, conditional_probability), uniforms[item])

                demand[item] = item_demand
                remaining_consumers -= item_demand
                remaining_probability -= purchase_probabilities[item]

        end

        return demand
end

# Demand history and purchase probabilities for one repetition and drift.
function generate_drift_data(drift, repetition, mixture_weights)

        # Multinomial chosen in each period.
        multinomial_rng = experiment_rng(repetition, multinomial_stream)
        multinomials = [sample(multinomial_rng, 1:number_of_multinomials, Weights(mixture_weights)) for _ in 1:history_length]

        # Uniforms for the demands and triangular innovations for the drift.
        uniforms = zeros(history_length, number_of_items)
        innovations = zeros(history_length-1, number_of_multinomials, number_of_items)
        next_innovations = zeros(number_of_future_samples, number_of_multinomials, number_of_items)
        for item in 1:number_of_items

                uniforms[:,item] = rand(experiment_rng(repetition, demand_stream, item = item), history_length)

                innovation_rng = experiment_rng(repetition, innovation_stream, item = item)
                innovations[:,:,item] = rand(innovation_rng, TriangularDist(-1, 1, 0), history_length-1, number_of_multinomials)
                next_innovations[:,:,item] = rand(innovation_rng, TriangularDist(-1, 1, 0), number_of_future_samples, number_of_multinomials)

        end

        purchase_probabilities = sample_initial_purchase_probabilities(repetition)
        initial_purchase_probabilities = deepcopy(purchase_probabilities)

        # Demand in each period followed by drift.
        demands = Vector{Float64}[]
        for period in 1:history_length

                push!(demands, sample_multinomial_demand(uniforms[period,:], purchase_probabilities[multinomials[period]]))

                if period < history_length
                        for multinomial in 1:number_of_multinomials
                                purchase_probabilities[multinomial] .+= drift*innovations[period,multinomial,:]
                                project_purchase_probabilities!(purchase_probabilities[multinomial])

                        end
                end
        end

        # Samples of one more period of drift.
        next_purchase_probabilities = zeros(number_of_future_samples, number_of_multinomials, number_of_items)
        for future_sample in 1:number_of_future_samples, multinomial in 1:number_of_multinomials
                next_purchase_probabilities[future_sample,multinomial,:] = project_purchase_probabilities!(purchase_probabilities[multinomial] + drift*next_innovations[future_sample,multinomial,:])

        end

        return demands, next_purchase_probabilities, initial_purchase_probabilities
end



# Cost of an order for a given demand.
function realized_multi_item_newsvendor_cost(order, demand, underage_costs, overage_costs)

        return sum(underage_costs[item]*max(demand[item] - order[item], 0) + overage_costs[item]*max(order[item] - demand[item], 0) for item in 1:number_of_items)
end

# Expected cost of an order for binomial demand (see test-expected-newsvendor-loss.jl).
function expected_newsvendor_cost_with_binomial_demand(order, purchase_probability, number_of_consumers, underage_cost, overage_cost)

        one_fewer_consumer_cdf = cdf(Binomial(number_of_consumers-1, purchase_probability), order-1)
        demand_cdf = cdf(Binomial(number_of_consumers, purchase_probability), order)

        expected_underage_cost = underage_cost*(number_of_consumers*purchase_probability*(1-one_fewer_consumer_cdf) - order*(1-demand_cdf))
        expected_overage_cost = overage_cost*(order*demand_cdf - number_of_consumers*purchase_probability*one_fewer_consumer_cdf)

        return expected_underage_cost + expected_overage_cost
end

# Expected cost of an order in the next period averaged over the samples.
function expected_multi_item_newsvendor_cost(order, next_purchase_probabilities, mixture_weights, underage_costs, overage_costs)

        cost = 0.0

        for future_sample in 1:number_of_future_samples, multinomial in 1:number_of_multinomials, item in 1:number_of_items
                cost += (1/number_of_future_samples)*mixture_weights[multinomial]*expected_newsvendor_cost_with_binomial_demand(order[item], next_purchase_probabilities[future_sample,multinomial,item], number_of_consumers, underage_costs[item], overage_costs[item])

        end

        return cost
end



# Weights for every weight parameter and number of past demands.
function weight_vector_table(compute_weights, parameters)

        return Dict(number_of_past_demands => [compute_weights(number_of_past_demands, parameter) for parameter in parameters] for number_of_past_demands in history_length-training_length:history_length)
end

# Average training cost of every ambiguity radius and weight parameter.
function average_training_costs(objective_value_and_order, ambiguity_radii, weight_vectors, demands, underage_costs, overage_costs)

        costs = zeros(length(ambiguity_radii), length(weight_vectors[history_length]))

        for number_of_past_demands in history_length-training_length:history_length-1

                objective_values_and_orders = _multi_item_newsvendor_grid(objective_value_and_order, ambiguity_radii, demands[1:number_of_past_demands], weight_vectors[number_of_past_demands], underage_costs, overage_costs)

                for weight_parameter_index in eachindex(weight_vectors[number_of_past_demands]), ambiguity_radius_index in eachindex(ambiguity_radii)
                        _, order = objective_values_and_orders[ambiguity_radius_index,weight_parameter_index]
                        costs[ambiguity_radius_index,weight_parameter_index] += realized_multi_item_newsvendor_cost(order, demands[number_of_past_demands+1], underage_costs, overage_costs)/training_length

                end
        end

        return costs
end

# Order for the next period using the parameters of smallest average training cost.
function train_and_test_order(objective_value_and_order, ambiguity_radii, weight_vectors, demands, underage_costs, overage_costs)

        ambiguity_radius_index, weight_parameter_index = Tuple(argmin(average_training_costs(objective_value_and_order, ambiguity_radii, weight_vectors, demands, underage_costs, overage_costs)))

        return objective_value_and_order(ambiguity_radii[ambiguity_radius_index], demands, weight_vectors[history_length][weight_parameter_index], underage_costs, overage_costs)[2]
end
