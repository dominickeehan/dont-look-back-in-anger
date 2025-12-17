# Change seed with job number.
# Output files get saved in same directory.
# Save in different files for each job number.
# Want each job to be under 8 hours.
# Write the chosen parameters to the file as well.

# This script has approximately < 8 hours runtime.

using Random, Statistics, StatsBase, Distributions
using ProgressBars, IterTools
using CSV


number_of_modes = 2
mixture_weights = [0.9, 0.1]
initial_demand_probabilities = [0.1, 0.5]
construct_drift_distribution(drift) = TriangularDist(-drift, drift, 0.0) # Same for each mode.
drifts = [1.00e-3, 1.79e-3, 3.16e-3, 5.62e-3, 1.00e-2, 1.79e-2, 3.16e-2, 5.62e-2, 1.00e-1, 1.79e-1, 3.16e-1] # exp10.(LinRange(log10(1),log10(10),5))
number_of_consumers = 1000.0
cu = 4.0 # Per-unit underage cost.
co = 1.0 # Per-unit overage cost.
include("weights.jl")
include("newsvendor-optimizations.jl")

number_of_jobs_per_drift = 1000
number_of_repetitions = 1 # Per each job.
history_length = 100 # 100
training_length = 30 # 30

# Open results file and wipe it to empty.
job_number = parse(Int64, ENV["PBS_ARRAY_INDEX"])
open("$job_number.csv", "w") do file; end
results_file = open("$job_number.csv", "a")
println(results_file, "drift, repetition index, method name, ambiguity radius, weight parameter, average training cost, doubling count, objective value, expected next period cost, time elapsed")

newsvendor_cost(order, demand) = cu*max(demand-order,0.0) + co*max(order-demand,0.0)

function expected_newsvendor_cost_with_binomial_demand(order, binomial_demand_probability)

    a = cdf(Binomial(number_of_consumers-1,binomial_demand_probability), order-1)
    b = cdf(Binomial(number_of_consumers,binomial_demand_probability), order)

    expected_underage_cost = cu * (number_of_consumers*binomial_demand_probability*(1-a) - order*(1-b))
    expected_overage_cost = co * (order*b - number_of_consumers*binomial_demand_probability*a)

    return expected_underage_cost + expected_overage_cost
end

Random.seed!(job_number%number_of_jobs_per_drift)
drift = drifts[ceil(Int,(job_number+1)/number_of_jobs_per_drift)]
drift_distribution = construct_drift_distribution(drift)

demand_sequences = [zeros(history_length) for _ in 1:number_of_repetitions]
final_demand_probabilities = [[zeros(number_of_modes) for _ in 1:1000] for _ in 1:number_of_repetitions]

for repetition_index in 1:number_of_repetitions
    local demand_probabilities = initial_demand_probabilities

    for t in 1:history_length
        demand_sequences[repetition_index][t] = 
            rand(MixtureModel(Binomial, [(number_of_consumers, demand_probabilities[i]) for i in 1:number_of_modes], mixture_weights))
        
        if t < history_length
            demand_probabilities = 
                [min(max(demand_probabilities[i] + rand(drift_distribution), 0.0), 1.0) for i in 1:number_of_modes]

        else
            for i in eachindex(final_demand_probabilities[repetition_index])
                final_demand_probabilities[repetition_index][i] =
                    [min(max(demand_probabilities[i] + rand(drift_distribution), 0.0), 1.0) for i in 1:number_of_modes]            

            end
        end
    end
end


function train_and_test(method, newsvendor_objective_value_and_order, ambiguity_radii, compute_weights, weight_parameters)

    precomputed_weights = 
        hcat([[zeros(t-1) for t in history_length-training_length+1:history_length+1] for weight_parameter_index in eachindex(weight_parameters)])

    println("Precomputing weights...")
    #Threads.@threads for weight_parameter_index in ProgressBar(eachindex(weight_parameters))
    for weight_parameter_index in ProgressBar(eachindex(weight_parameters))
        for t in history_length-training_length+1:history_length+1
            precomputed_weights[weight_parameter_index][t-(history_length-training_length)] = compute_weights(t-1, weight_parameters[weight_parameter_index])
        
        end
    end

    println("Training and testing...")
    for repetition_index in 1:number_of_repetitions

        start_time = time()

        training_costs = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in history_length-training_length+1:history_length]
        objective_values = zeros((length(ambiguity_radii),length(weight_parameters)))
        expected_next_period_costs = zeros((length(ambiguity_radii),length(weight_parameters)))
        doubling_counts = zeros((length(ambiguity_radii),length(weight_parameters)))


        #Threads.@threads for ambiguity_radius_index in ProgressBar(eachindex(ambiguity_radii))
        for ambiguity_radius_index in ProgressBar(eachindex(ambiguity_radii))
            for weight_parameter_index in eachindex(weight_parameters)   
                for t in history_length-training_length+1:history_length
                    local weights = precomputed_weights[weight_parameter_index][t-(history_length-training_length)]
                    local demand_samples = demand_sequences[repetition_index][1:t-1]

                    local _, order, _ = 
                        newsvendor_objective_value_and_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights, 0)
                    
                    training_costs[t-(history_length-training_length)][ambiguity_radius_index, weight_parameter_index] = newsvendor_cost(order, demand_sequences[repetition_index][t])

                end
            end
        end

        #Threads.@threads for ambiguity_radius_index in ProgressBar(eachindex(ambiguity_radii))
        for ambiguity_radius_index in ProgressBar(eachindex(ambiguity_radii))
            for weight_parameter_index in eachindex(weight_parameters)
                local weights = precomputed_weights[weight_parameter_index][end]
                local demand_samples = demand_sequences[repetition_index][1:end]

                local objective_values[ambiguity_radius_index, weight_parameter_index], order, doubling_counts[ambiguity_radius_index, weight_parameter_index] = 
                    newsvendor_objective_value_and_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights, 0)
                
                expected_next_period_costs[ambiguity_radius_index, weight_parameter_index] = 
                    mean([sum(mixture_weights[j]*expected_newsvendor_cost_with_binomial_demand(order, final_demand_probabilities[repetition_index][i][j]) for j in 1:number_of_modes) for i in eachindex(final_demand_probabilities[repetition_index])])

            end
        end

        average_training_costs = mean(training_costs)

        time_elapsed = time() - start_time

        for ambiguity_radius_index in eachindex(ambiguity_radii)
            for weight_parameter_index in eachindex(weight_parameters)
                ambiguity_radius = ambiguity_radii[ambiguity_radius_index]
                weight_parameter = weight_parameters[weight_parameter_index]
                average_training_cost = average_training_costs[ambiguity_radius_index, weight_parameter_index]
                doubling_count = doubling_counts[ambiguity_radius_index, weight_parameter_index]
                objective_value = objective_values[ambiguity_radius_index, weight_parameter_index]
                expected_next_period_cost = expected_next_period_costs[ambiguity_radius_index, weight_parameter_index]

                println(results_file, "$drift, $repetition_index, $method, $ambiguity_radius, $weight_parameter, $average_training_cost, $doubling_count, $objective_value, $expected_next_period_cost, $time_elapsed")

            end
        end
    end
end

LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))
ε = number_of_consumers*unique([0.0; LinRange(1.0e-3,1.0e-2,10); LinRange(1.0e-2,1.0e-1,10); LinRange(1.0e-1,1.0e-0,10)])
s = unique(round.(Int, LogRange(1,history_length,30)))
α = [0.0; LogRange(1.0e-4,1.0e0,30)]
ρ╱ε = [0.0; LogRange(1.0e-4,1.0e0,30)]
intersection_ε = number_of_consumers*unique([LinRange(1.0e-3,1.0e-2,10); LinRange(1.0e-2,1.0e-1,10); LinRange(1.0e-1,1.0e-0,10)])
intersection_ρ╱ε = [0.0; LogRange(1.0e-4,1.0e0,30)]

println("Training and testing stochastic optimization methods...")
train_and_test("SAA (\$ε=0\$)", SO_newsvendor_objective_value_and_order, [0.0], windowing_weights, [history_length])
train_and_test("Smoothing (\$ε=0\$)", SO_newsvendor_objective_value_and_order, [0.0], smoothing_weights, α)

println("Training and testing W2 distributionally robust optimization methods...")
train_and_test("Intersection", REMK_intersection_W2_DRO_newsvendor_objective_value_and_order, intersection_ε, REMK_intersection_weights, intersection_ρ╱ε)
train_and_test("Weighted", W2_DRO_newsvendor_objective_value_and_order, ε, W2_weights, ρ╱ε)

close(results_file)
