# Change seed with job number.
# Output files get saved in same directory.
# Save in different files for each job number.
# Want each job to be under 8 hours.
# Write the chosen parameters to the file as well.

# Approximately < 8 hours runtime.

using Random, Statistics, StatsBase, Distributions
using ProgressBars, IterTools
using CSV

include("weights.jl")
include("newsvendor-optimizations.jl")

job_number = parse(Int64, ENV["PBS_ARRAY_INDEX"])

open("$job_number.csv", "w") do file; end

results_file = open("$job_number.csv", "a")

repetitions = 1 # (1000 total jobs for each u ∈ U.)
history_length = 100 # 100
training_length = 30 # 30


initial_demand_probability = 0.1 # 0.1


newsvendor_loss(order, demand) = Cu*max(demand-order,0) + Co*max(order-demand,0)

function expected_newsvendor_loss(order, demand_probability)

    a = cdf(Binomial(D-1,demand_probability), order-1)
    b = cdf(Binomial(D,demand_probability), order)

    expected_underage_cost = Cu * (D*demand_probability*(1-a) - order*(1-b))
    expected_overage_cost = Co * (order*b - D*demand_probability*a)

    return expected_underage_cost + expected_overage_cost

end

U = [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2]

Random.seed!(job_number%1000)

u = U[ceil(Int,(job_number+1)/1000)]

shift_distribution = Uniform(-u,u)

demand_sequences = [zeros(history_length) for _ in 1:repetitions]
final_demand_probabilities = [zeros(10000) for _ in 1:repetitions]

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


function train_and_test(newsvendor_value_and_order, ambiguity_radii, compute_weights, weight_parameters)

    costs = zeros(repetitions)

    precomputed_weights = 
        hcat([[zeros(t-1) for t in history_length-training_length+1:history_length] for weight_parameter_index in eachindex(weight_parameters)])

    println("Precomputing weights...")
    #Threads.@threads for weight_parameter_index in ProgressBar(eachindex(weight_parameters))
    for weight_parameter_index in ProgressBar(eachindex(weight_parameters))
        for t in history_length-training_length+1:history_length
            precomputed_weights[weight_parameter_index][t-(history_length-training_length)] = compute_weights(t-1, weight_parameters[weight_parameter_index])
        
        end
    end

    println("Training and testing...")
    for repetition in ProgressBar(1:repetitions)

        start_time = time()

        training_costs = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in history_length-training_length+1:history_length]

        #Threads.@threads for ambiguity_radius_index in eachindex(ambiguity_radii)
        for ambiguity_radius_index in eachindex(ambiguity_radii)
            for weight_parameter_index in eachindex(weight_parameters)   
                for t in history_length-training_length+1:history_length                
                    local weights = precomputed_weights[weight_parameter_index][t-(history_length-training_length)]
                    local demand_samples = demand_sequences[repetition][1:t-1]

                    local _, order, _ = newsvendor_value_and_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights, 0)
                    
                    training_costs[t-(history_length-training_length)][ambiguity_radius_index, weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][t])

                end
            end
        end

        ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(training_costs)))
        weights = compute_weights(history_length, weight_parameters[weight_parameter_index])
        demand_samples = demand_sequences[repetition][1:history_length]

        value, order, doubling_count = newsvendor_value_and_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights, 0)

        costs[repetition] = mean([expected_newsvendor_loss(order, final_demand_probabilities[repetition][i]) for i in eachindex(final_demand_probabilities[repetition])])
        realised_cost = costs[repetition]

        tested_ambiguity_radius = ambiguity_radii[ambiguity_radius_index]
        tested_weight_parameter = weight_parameters[weight_parameter_index]

        time_elapsed = time() - start_time

        println(results_file, "$u, $tested_ambiguity_radius, $tested_weight_parameter, $doubling_count, $value, $realised_cost, $time_elapsed")

    end

    μ = mean(costs)
    σ = sem(costs)
    display("$μ ± $σ")
    
end

ε = [0; LinRange(1e0,1e1,10); LinRange(2e1,1e2,9); LinRange(2e2,1e3,9); LinRange(2e3,1e4,9); LinRange(2e4,1e5,9)]
s = [round.(Int, LinRange(1,10,10)); round.(Int, LinRange(12,30,10)); round.(Int, LinRange(33,60,10)); round.(Int, LinRange(64,100,10))]

LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))

α = [0; LogRange(1e-4,1e0,39)]
ϱ╱ε = [0; LogRange(1e-4,1e0,39)]

println("SO...")

train_and_test(SO_newsvendor_value_and_order, [0], windowing_weights, [history_length])
train_and_test(SO_newsvendor_value_and_order, [0], windowing_weights, s)
train_and_test(SO_newsvendor_value_and_order, [0], smoothing_weights, α)

println("W1...")

train_and_test(W1_newsvendor_value_and_order, ε, windowing_weights, [history_length])
train_and_test(W1_newsvendor_value_and_order, ε, windowing_weights, s)
train_and_test(W1_newsvendor_value_and_order, ε, smoothing_weights, α)
train_and_test(W1_newsvendor_value_and_order, ε, W1_concentration_weights, ϱ╱ε)

println("W2...")

train_and_test(W2_newsvendor_value_and_order, ε, windowing_weights, [history_length])
train_and_test(W2_newsvendor_value_and_order, ε, windowing_weights, s)
train_and_test(W2_newsvendor_value_and_order, ε, smoothing_weights, α)
train_and_test(W2_newsvendor_value_and_order, ε, W2_concentration_weights, ϱ╱ε)

ε = [LinRange(1e2,1e3,10); LinRange(2e3,1e4,9); LinRange(2e4,1e5,9); LinRange(2e5,1e6,9); LinRange(2e6,1e7,9)]
ϱ╱ε = [0; LogRange(1e-4,1e2,39)]

println("Intersections...")

train_and_test(REMK_intersection_W2_newsvendor_value_and_order, ε, REMK_intersection_weights, ϱ╱ε)

close(results_file)