using Random, Statistics, StatsBase, Distributions
using ProgressBars, IterTools

include("weights.jl")
include("newsvendor-optimizations.jl")

repetitions = 10
history_length = 70

function expected_newsvendor_cost(order, demand_probability)

    a = cdf(Binomial(D-1,demand_probability), order-1)
    b = cdf(Binomial(D,demand_probability), order)

    expected_underage_cost = Cu * (D*demand_probability*(1-a) - order*(1-b))
    expected_overage_cost = Co * (order*b - D*demand_probability*a)

    return expected_underage_cost + expected_overage_cost

end

Random.seed!(42)
u = 2.5e-3 # [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2]
shift_distribution = Uniform(-u,u)

demand_sequences = [zeros(history_length) for _ in 1:repetitions]
final_demand_probabilities = [zeros(100) for _ in 1:repetitions]

for repetition in 1:repetitions
    local demand_probability = initial_demand_probability

    for t in 1:history_length
        demand_sequences[repetition][t] = rand(Binomial(D, demand_probability))
        
        if t < history_length
            demand_probability = min(max(demand_probability + rand(shift_distribution), 0), 1)

        else
            for i in eachindex(final_demand_probabilities[repetition])
                final_demand_probabilities[repetition][i] = min(max(demand_probability + rand(shift_distribution), 0), 1)
        
            end
        end
    end
end

function parameter_fit(newsvendor_objective_value_and_order, ambiguity_radii, compute_weights, weight_parameters)

    costs = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in 1:repetitions]
    doubling_count = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in 1:repetitions]

    precomputed_weights = [zeros(history_length) for weight_parameter_index in eachindex(weight_parameters)]

    println("Precomputing weights...")
    Threads.@threads for weight_parameter_index in ProgressBar(eachindex(weight_parameters))
        precomputed_weights[weight_parameter_index] = compute_weights(history_length, weight_parameters[weight_parameter_index])

    end

    println("Parameter fitting...")
    Threads.@threads for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
        for repetition in 1:repetitions
            local weights = precomputed_weights[weight_parameter_index]
            local demand_samples = demand_sequences[repetition][1:history_length]

            local _, order, doubling_count[repetition][ambiguity_radius_index, weight_parameter_index] = newsvendor_objective_value_and_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights, 0)

            costs[repetition][ambiguity_radius_index, weight_parameter_index] = 
                mean([expected_newsvendor_cost(order, final_demand_probabilities[repetition][i]) for i in eachindex(final_demand_probabilities[repetition])])

        end
    end

    display(compute_weights)

    ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][ambiguity_radius_index, weight_parameter_index] for repetition in 1:repetitions]
    μ = mean(minimal_costs)
    σ = sem(minimal_costs)
    println("Ex-post minimal average cost: $μ ± $σ")
    
    optimal_ambiguity_radius = ambiguity_radii[ambiguity_radius_index]
    optimal_weight_parameter = weight_parameters[weight_parameter_index]
    println("Optimal ambiguity radius: $optimal_ambiguity_radius")
    println("Optimal weight parameter: $optimal_weight_parameter")

    optimal_doubling_count = mean([doubling_count[repetition][ambiguity_radius_index, weight_parameter_index] for repetition in 1:repetitions])
    println("Optimal doubling count: $optimal_doubling_count")

end

LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))

ε = [0; LinRange(1e-1,1e0,10); LinRange(2e0,1e1,9); LinRange(2e1,1e2,9); LinRange(2e2,1e3,9);]
s = unique(round.(Int, LogRange(1,100,40)))

α = [0; LogRange(1e-4,1e0,40)]
ρ╱ε = [0; LogRange(1e-4,1e0,40)]

#parameter_fit(SO_newsvendor_objective_value_and_order, [0], windowing_weights, [history_length])
#parameter_fit(SO_newsvendor_objective_value_and_order, [0], smoothing_weights, α)

#parameter_fit(W1_newsvendor_objective_value_and_order, ε, windowing_weights, [history_length])
#parameter_fit(W1_newsvendor_objective_value_and_order, ε, windowing_weights, s)
#parameter_fit(W1_newsvendor_objective_value_and_order, ε, smoothing_weights, α)
#parameter_fit(W1_newsvendor_objective_value_and_order, ε, W1_weights, ρ╱ε)

#parameter_fit(W1_newsvendor_objective_value_and_order, 0, windowing_weights, s)
#parameter_fit(W1_newsvendor_objective_value_and_order, 0, smoothing_weights, α)
#parameter_fit(W1_newsvendor_objective_value_and_order, 0, W1_weights, ρ╱ε)

#parameter_fit(W2_newsvendor_objective_value_and_order, ε, windowing_weights, [history_length])
#parameter_fit(W2_newsvendor_objective_value_and_order, ε, windowing_weights, s)
#parameter_fit(W2_newsvendor_objective_value_and_order, ε, smoothing_weights, α)
parameter_fit(W2_newsvendor_objective_value_and_order, ε, W2_weights, ρ╱ε)

ε = [LinRange(1e0,1e1,10); LinRange(2e1,1e2,9); LinRange(2e2,1e3,9); LinRange(2e3,1e4,9);]
ρ╱ε = [0; LogRange(1e-4,1e0,40)]

parameter_fit(REMK_intersection_W2_newsvendor_objective_value_and_order, ε, REMK_intersection_weights, ρ╱ε)


