# Change seed with job number.
# Output files get saved in same directory. 
# Save in different files for each job number. 
# Want each job to be under 9 hours. 
# Write the chosen parameters to the file as well.

using Random, Statistics, StatsBase, Distributions
using JuMP, MathOptInterface, Gurobi
using ProgressBars, IterTools
using CSV

number_of_consumers = 10000
D = number_of_consumers

Cu = 3 # Cost of underage.
Co = 1 # Cost of overage.

newsvendor_loss(x,ξ) = Cu*max(ξ-x,0) + Co*max(x-ξ,0)
newsvendor_order(ε, ξ, weights) = quantile(ξ, Weights(weights), Cu/(Co+Cu))

env = Gurobi.Env() 
GRBsetintparam(env, "OutputFlag", 0)
GRBsetintparam(env, "BarHomogeneous", 1)
optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

include("weights.jl")

Random.seed!(42)

shift_distribution = Uniform(-0.001,0.001)

initial_demand_probability = 0.1

repetitions = 10000
history_length = 100
training_length = 10

demand_sequences = [zeros(history_length+1) for _ in 1:repetitions]
for repetition in 1:repetitions
    demand_probability = initial_demand_probability
    for t in 1:history_length+1
        demand_sequences[repetition][t] = rand(Binomial(number_of_consumers, demand_probability))
        demand_probability = min(max(demand_probability + rand(shift_distribution), 0), 1.0)
    end
end

function train_and_test(ambiguity_radii, compute_weights, weight_parameters)

    costs = zeros(repetitions)

    ambiguity_radii_to_test = zeros(repetitions)
    weight_parameters_to_test = zeros(repetitions)

    precomputed_weights = stack([[[zeros(t-1) for t in history_length-training_length+1:history_length] for ambiguity_radius_index in eachindex(ambiguity_radii)] for weight_parameter_index in eachindex(weight_parameters)])

    #println("precomputing_weights...")

    Threads.@threads for (ambiguity_radius_index, weight_parameter_index, t) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters), history_length-training_length+1:history_length)))
        precomputed_weights[ambiguity_radius_index, weight_parameter_index][t-(history_length-training_length)] = compute_weights(t-1, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index])
    end

    #println("training and testing SP method...")

    Threads.@threads for repetition in ProgressBar(1:repetitions)
        
        training_costs = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in history_length-training_length+1:history_length]
        for ambiguity_radius_index in eachindex(ambiguity_radii)
            for weight_parameter_index in eachindex(weight_parameters)
                
                for t in history_length-training_length+1:history_length
                
                    weights = precomputed_weights[ambiguity_radius_index, weight_parameter_index][t-(history_length-training_length)]

                    demand_samples = demand_sequences[repetition][1:t-1]
                    order = newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
                    training_costs[t-(history_length-training_length)][ambiguity_radius_index, weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][t])
                end
            end
        end

        ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(training_costs)))
        weights = compute_weights(history_length, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index])
        demand_samples = demand_sequences[repetition][1:history_length]
        order = newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)

        costs[repetition] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])

    end

    μ = mean(costs)
    σ = sem(costs)
    println("$μ ± $σ")
end

windowing_parameters = round.(Int, LinRange(1,history_length,history_length))
smoothing_parameters = [LinRange(0.0001,0.001,10); LinRange(0.002,0.01,9); LinRange(0.02,1.0,99)]
ambiguity_radii = [history_length]
shift_bound_parameters = LinRange(1,history_length,10)

println("SAA:")
train_and_test([0], windowing_weights, [history_length])
println("Windowing:")
train_and_test([0], windowing_weights, windowing_parameters)
println("Smoothing:")
train_and_test([0], smoothing_weights, smoothing_parameters)
println("Concentration:")
train_and_test(ambiguity_radii, W₁_concentration_weights, shift_bound_parameters)