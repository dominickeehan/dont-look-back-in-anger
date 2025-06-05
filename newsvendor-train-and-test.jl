# Change seed with job number.
# Output files get saved in same directory.
# Save in different files for each job number.
# Want each job to be under 8 hours.
# Write the chosen parameters to the file as well.

# Approximately 7 hours runtime.

using Random, Statistics, StatsBase, Distributions
using ProgressBars, IterTools
using CSV

job_number = parse(Int64, ENV["PBS_ARRAY_INDEX"])

open("$job_number.csv", "w") do file; end

results_file = open("$job_number.csv", "a")

repetitions = 1 # 1000 in total with all other jobs per U, 7000 in total.
history_length = 100 # 100
training_length = 30 # 30


initial_demand_probability = 0.1 # 0.1


include("weights.jl")

include("newsvendor-optimizations.jl")

newsvendor_loss(order, demand) = Cu*max(demand-order,0) + Co*max(order-demand,0)

function expected_newsvendor_loss(order, demand_probability)

    expected_underage = D*demand_probability*(1-cdf(Binomial(D-1,demand_probability),order-2)) -
        order*(1-cdf(Binomial(D,demand_probability),order-1))

    expected_overage = order*cdf(Binomial(D,demand_probability),order) -
        D*demand_probability*(cdf(Binomial(D-1,demand_probability),order-1))

    return Cu*expected_underage + Co*expected_overage

end

Us = [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2]

Random.seed!(job_number%1000)

U = Us[ceil(Int,(job_number+1)/1000)]

shift_distribution = Uniform(-U,U)

demand_sequences = [zeros(history_length+1) for _ in 1:repetitions]
demand_probability = [zeros(history_length+1) for _ in 1:repetitions]

for repetition in 1:repetitions
    demand_probability[repetition][1] = initial_demand_probability

    for t in 1:history_length+1
        demand_sequences[repetition][t] = rand(Binomial(D, demand_probability[repetition][t]))
        
        if t < history_length+1
            demand_probability[repetition][t+1] = min(max(demand_probability[repetition][t] + rand(shift_distribution), 0.0), 1.0)
        end
    end
end


function train_and_test(newsvendor_value_and_order, ambiguity_radii, compute_weights, weight_parameters)

    costs = zeros(repetitions)

    precomputed_weights = hcat([[zeros(t-1) for t in history_length-training_length+1:history_length] for weight_parameter_index in eachindex(weight_parameters)])

    println("Precomputing_weights...")
    #Threads.@threads for weight_parameter_index in ProgressBar(eachindex(weight_parameters))
    for weight_parameter_index in ProgressBar(eachindex(weight_parameters))
        for t in history_length-training_length+1:history_length
            precomputed_weights[weight_parameter_index][t-(history_length-training_length)] = 
                compute_weights(t-1, weight_parameters[weight_parameter_index])
        
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
                    local weights = 
                        precomputed_weights[weight_parameter_index][t-(history_length-training_length)]

                    local demand_samples = demand_sequences[repetition][1:t-1]
                    local _, order = newsvendor_value_and_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
                    training_costs[t-(history_length-training_length)][ambiguity_radius_index, weight_parameter_index] = 
                        newsvendor_loss(order, demand_sequences[repetition][t])

                end
            end
        end

        ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(training_costs)))
        weights = compute_weights(history_length, weight_parameters[weight_parameter_index])
    
        demand_samples = demand_sequences[repetition][1:history_length]
        value, order = newsvendor_value_and_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)

        #costs[repetition] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])
        costs[repetition] = expected_newsvendor_loss(order, demand_probability[repetition][history_length+1])
        realised_cost = costs[repetition]

        tested_ambiguity_radius = ambiguity_radii[ambiguity_radius_index]
        tested_weight_parameter = weight_parameters[weight_parameter_index]

        time_elapsed = time() - start_time

        println(results_file, "$U, $tested_ambiguity_radius, $tested_weight_parameter, $value, $realised_cost, $time_elapsed")

    end

    #μ = mean(costs)
    #σ = sem(costs)
    #display("$μ ± $σ")
    
end

ε = [[0]; LinRange(1e0,1e1,10); LinRange(2e1,1e2,9); LinRange(2e2,1e3,9); LinRange(2e3,1e4,9); LinRange(2e4,1e5,9)]
s = [round.(Int, LinRange(1,10,10)); round.(Int, LinRange(12,30,10)); round.(Int, LinRange(33,60,10)); round.(Int, LinRange(64,100,10));]
α = [[0]; LinRange(1e-4,1e-3,10); LinRange(2e-3,1e-2,9); LinRange(2e-2,1e-1,9); LinRange(2e-1,1e0,9)]
ϱ_divided_by_ε = [[0]; LinRange(1e-4,1e-3,10); LinRange(2e-3,1e-2,9); LinRange(2e-2,1e-1,9); LinRange(2e-1,1e0,9)]

println("SO...")

train_and_test(SO_newsvendor_value_and_order, 0, windowing_weights, [history_length])
train_and_test(SO_newsvendor_value_and_order, 0, windowing_weights, s)
train_and_test(SO_newsvendor_value_and_order, 0, smoothing_weights, α)

println("W1...")

train_and_test(W1_newsvendor_value_and_order, ε, windowing_weights, [history_length])
train_and_test(W1_newsvendor_value_and_order, ε, windowing_weights, s)
train_and_test(W1_newsvendor_value_and_order, ε, smoothing_weights, α)
train_and_test(W1_newsvendor_value_and_order, ε, W1_concentration_weights, ϱ_divided_by_ε)


println("W2...")

train_and_test(W2_newsvendor_value_and_order, ε, windowing_weights, [history_length])
train_and_test(W2_newsvendor_value_and_order, ε, windowing_weights, s)
train_and_test(W2_newsvendor_value_and_order, ε, smoothing_weights, α)
train_and_test(W2_newsvendor_value_and_order, ε, W2_concentration_weights, ϱ_divided_by_ε)

ε = [LinRange(1e2,1e3,10); LinRange(2e3,1e4,9); LinRange(2e4,1e5,9); LinRange(2e5,1e6,9); LinRange(2e6,1e7,9)]
ϱ_divided_by_ε = [[0]; LinRange(1e-4,1e-3,10); LinRange(2e-3,1e-2,9); LinRange(2e-2,1e-1,9); LinRange(2e-1,1e0,9)]

train_and_test(REMK_intersection_W2_newsvendor_value_and_order, ε, REMK_intersection_weights, ϱ_divided_by_ε)

close(results_file)