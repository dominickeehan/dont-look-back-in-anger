# Change seed with job number.
# Output files get saved in same directory.
# Save in different files for each job number.
# Want each job to be under 8 hours.
# Write the chosen parameters to the file as well.

# Approx. 7.5 hour runtime.

using Random, Statistics, StatsBase, Distributions
using JuMP, MathOptInterface, Gurobi
using ProgressBars, IterTools
using CSV

job_number = 42 #parse(Int64, ENV["PBS_ARRAY_INDEX"])

open("$job_number.csv", "w") do file; end

results_file = open("$job_number.csv", "a")

repetitions = 1 # 1000 (in total with all other jobs)
history_length = 100 # 100
training_length = 30 # 30

number_of_consumers = 10000 # 10000
D = number_of_consumers

initial_demand_probability = 0.1 # 0.1

Cu = 4 # 4 # Per-unit underage cost.
Co = 1 # 1 # Per-unit overage cost.

newsvendor_loss(order, demand) = Cu*max(demand-order,0) + Co*max(order-demand,0)

function expected_newsvendor_loss(order, demand_probability)

    expected_underage = D*demand_probability*(1-cdf(Binomial(D-1,demand_probability),order-2)) -
        order*(1-cdf(Binomial(D,demand_probability),order-1))

    expected_overage = order*cdf(Binomial(D,demand_probability),order) -
        D*demand_probability*(cdf(Binomial(D-1,demand_probability),order-1))

    return Cu*expected_underage + Co*expected_overage

end

include("weights.jl")

env = Gurobi.Env() 
GRBsetintparam(env, "OutputFlag", 0)
GRBsetintparam(env, "BarHomogeneous", 1)
optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

include("newsvendor-orders.jl")

for U in [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01] # [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01]

    Random.seed!(job_number)
    
    shift_distribution = Uniform(-U,U)

    demand_sequences = [zeros(history_length+1) for _ in 1:repetitions]
    demand_probability = [zeros(history_length+1) for _ in 1:repetitions]

    for repetition in 1:repetitions
        demand_probability[repetition][1] = initial_demand_probability

        for t in 1:history_length+1
            demand_sequences[repetition][t] = rand(Binomial(number_of_consumers, demand_probability[repetition][t]))
            
            if t < history_length+1
                demand_probability[repetition][t+1] = min(max(demand_probability[repetition][t] + rand(shift_distribution), 0.0), 1.0)
            end
        end
    end


    function train_and_test(newsvendor_order, ambiguity_radii, compute_weights, weight_parameters)

        costs = zeros(repetitions)
    
        #precomputed_weights = stack([[[zeros(t-1) for t in history_length-training_length+1:history_length] for ambiguity_radius_index in eachindex(ambiguity_radii)] for weight_parameter_index in eachindex(weight_parameters)])
        precomputed_weights = hcat([[[zeros(t-1) for t in history_length-training_length+1:history_length] for ambiguity_radius_index in eachindex(ambiguity_radii)] for weight_parameter_index in eachindex(weight_parameters)]...)
    
        println("Precomputing_weights...")
    
        #Threads.@threads for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
        for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
            for t in history_length-training_length+1:history_length
                precomputed_weights[ambiguity_radius_index, weight_parameter_index][t-(history_length-training_length)] = 
                    compute_weights(t-1, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index])
            
            end
        end
    
        println("Training and testing...")
    
        #Threads.@threads for repetition in ProgressBar(1:repetitions)
        for repetition in ProgressBar(1:repetitions)
    
            start_time = time()
    
            training_costs = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in history_length-training_length+1:history_length]
    
            for ambiguity_radius_index in eachindex(ambiguity_radii)
                for weight_parameter_index in eachindex(weight_parameters)                
                    for t in history_length-training_length+1:history_length                
                        weights = 
                            precomputed_weights[ambiguity_radius_index, weight_parameter_index][t-(history_length-training_length)]
    
                        demand_samples = demand_sequences[repetition][1:t-1]
                        order = newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
                        training_costs[t-(history_length-training_length)][ambiguity_radius_index, weight_parameter_index] = 
                            newsvendor_loss(order, demand_sequences[repetition][t])
    
                    end
                end
            end
    
            ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(training_costs)))
        
            weights = 
                compute_weights(history_length, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index])
        
            demand_samples = demand_sequences[repetition][1:history_length]
            order = newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
    
            #costs[repetition] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])
            costs[repetition] = expected_newsvendor_loss(order, demand_probability[repetition][history_length+1])
            realised_cost = costs[repetition]

            tested_ambiguity_radius = ambiguity_radii[ambiguity_radius_index]
            tested_weight_parameter = weight_parameters[weight_parameter_index]
    
            time_elapsed = time() - start_time
    
            println(results_file, "$U, $realised_cost, $tested_ambiguity_radius, $tested_weight_parameter, $time_elapsed")
    
        end
    
        #μ = mean(costs)
        #σ = sem(costs)
        #display("$μ ± $σ")
        
    end

    s = round.(Int, LinRange(1,history_length,34))
    α = [LinRange(0.0001,0.001,10); LinRange(0.002,0.01,9); LinRange(0.02,0.1,9); LinRange(0.2,1.0,9)]
    ε = [1000]
    ϱ = [[0]; LinRange(0.1,1,10); LinRange(2,10,9); LinRange(20,100,9); LinRange(200,1000,9)]

    train_and_test(W1_newsvendor_order, [0], windowing_weights, [history_length])
    train_and_test(W1_newsvendor_order, [0], windowing_weights, s)
    train_and_test(W1_newsvendor_order, [0], smoothing_weights, α)
    train_and_test(W1_newsvendor_order, ε, W1_weights, ϱ)


    ε = [LinRange(1,10,10); LinRange(20,100,9); LinRange(200,1000,9); LinRange(2000,10000,9)]
    s = round.(Int, LinRange(1,history_length,34))
    α = [LinRange(0.0001,0.001,10); LinRange(0.002,0.01,9); LinRange(0.02,0.1,9); LinRange(0.2,1.0,9)]
    ϱ = [[0]; LinRange(0.1,1,10); LinRange(2,10,9); LinRange(20,100,9); LinRange(200,1000,9)]

    train_and_test(W2_newsvendor_order, ε, windowing_weights, [history_length])
    train_and_test(W2_newsvendor_order, ε, windowing_weights, s)
    train_and_test(W2_newsvendor_order, ε, smoothing_weights, α)
    train_and_test(W2_newsvendor_order, ε, W2_weights, ϱ)


    ε = [LinRange(100,1000,10); LinRange(2000,10000,9); LinRange(20000,100000,9); LinRange(200000,1000000,9)]
    ϱ = [LinRange(1,10,10); LinRange(20,100,9); LinRange(200,1000,9); LinRange(2000,10000,9)]

    train_and_test(REMK_intersection_based_W2_newsvendor_order, ε, REMK_intersection_ball_radii, ϱ)

end

close(results_file)