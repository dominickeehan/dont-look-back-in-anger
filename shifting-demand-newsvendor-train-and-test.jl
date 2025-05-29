# Change seed with job number.
# Output files get saved in same directory. 
# Save in different files for each job number. 
# Want each job to be under 9 hours. 
# Write the chosen parameters to the file as well.

using Random, Statistics, StatsBase, Distributions
using JuMP, MathOptInterface, Gurobi
using ProgressBars, IterTools
using CSV

include("weights.jl")

job_number = 42 #parse(Int64, ENV["PBS_ARRAY_INDEX"])

Random.seed!(job_number)

open("$job_number.csv", "w") do file; end

results_file = open("$job_number.csv", "a")

repetitions = 1000
history_length = 100
training_length = 30

number_of_consumers = 10000
D = number_of_consumers

shift_distribution = Uniform(-0.0001,0.0001)

initial_demand_probability = 0.1

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

Cu = 4 # Per-unit underage cost.
Co = 1 # Per-unit overage cost.

newsvendor_loss(order, demand) = Cu*max(demand-order,0) + Co*max(order-demand,0)

function expected_newsvendor_loss(order, demand_probability)

    expected_underage = D*demand_probability*(1-cdf(Binomial(D-1,demand_probability),order-2)) 
        - x*(1-cdf(Binomial(D,demand_probability),order-1))

    expected_overage = order*cdf(Binomial(D,demand_probability),order) 
        - D*demand_probability*(cdf(Binomial(D-1,demand_probability),order-1))

    return Cu*expected_underage + Co*expected_overage

end

W1_newsvendor_order(ε, demands, weights) = quantile(demands, Weights(weights), Cu/(Co+Cu))

function train_and_test(newsvendor_order, ambiguity_radii, compute_weights, weight_parameters)

    costs = zeros(repetitions)

    precomputed_weights = stack([[[zeros(t-1) for t in history_length-training_length+1:history_length] for ambiguity_radius_index in eachindex(ambiguity_radii)] for weight_parameter_index in eachindex(weight_parameters)])
    #precomputed_weights = hcat([[[zeros(t-1) for t in history_length-training_length+1:history_length] for ambiguity_radius_index in eachindex(ambiguity_radii)] for weight_parameter_index in eachindex(weight_parameters)]...)

    println("Precomputing_weights...")

    Threads.@threads for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
    #for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
        for t in history_length-training_length+1:history_length
            precomputed_weights[ambiguity_radius_index, weight_parameter_index][t-(history_length-training_length)] = compute_weights(t-1, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index])
        
        end
    end

    println("Training and testing W1 method...")

    Threads.@threads for repetition in ProgressBar(1:repetitions)
    #for repetition in ProgressBar(1:repetitions)

        start_time = time()

        training_costs = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in history_length-training_length+1:history_length]

        for ambiguity_radius_index in eachindex(ambiguity_radii)
            for weight_parameter_index in eachindex(weight_parameters)                
                for t in history_length-training_length+1:history_length                
                    weights = precomputed_weights[ambiguity_radius_index, weight_parameter_index][t-(history_length-training_length)]

                    demand_samples = demand_sequences[repetition][1:t-1]
                    order = newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
                    training_costs[t-(history_length-training_length)][ambiguity_radius_index, weight_parameter_index] = 
                        newsvendor_loss(order, demand_sequences[repetition][t])

                end
            end
        end

        ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(training_costs)))
    
        weights = compute_weights(history_length, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index])
    
        demand_samples = demand_sequences[repetition][1:history_length]
        order = newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)

        costs[repetition] = expected_newsvendor_loss(order, demand_probability[repetition][history_length+1])
        #costs[repetition] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])
        
        tested_ambiguity_radius = ambiguity_radii[ambiguity_radius_index]
        tested_weight_parameter = weight_parameters[weight_parameter_index]

        time_elapsed = time() - start_time

        #println(results_file, "$costs[repetition], $tested_ambiguity_radius, $tested_weight_parameter, time_elapsed")

    end

    μ = mean(costs)
    σ = sem(costs)
    display("$μ ± $σ")
    
end

s = round.(Int, LinRange(1,history_length,34))
α = [LinRange(0.0001,0.001,10); LinRange(0.002,0.01,9); LinRange(0.02,0.1,9); LinRange(0.2,1.0,9)]
ε = [1000]
ϱ = [[0]; LinRange(0.1,1,10); LinRange(2,10,9); LinRange(20,100,9); LinRange(200,1000,9)]

train_and_test(W1_newsvendor_order, [0], windowing_weights, [history_length])
train_and_test(W1_newsvendor_order, [0], windowing_weights, s)
train_and_test(W1_newsvendor_order, [0], smoothing_weights, α)
train_and_test(W1_newsvendor_order, ε, W1_weights, ϱ)


env = Gurobi.Env() 
GRBsetintparam(env, "OutputFlag", 0)
GRBsetintparam(env, "BarHomogeneous", 1)
optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

function W2_newsvendor_order(ε, demands, weights) 

    demands = demands[weights .>= 1e-3]
    weights = weights[weights .>= 1e-3]
    weights .= weights/sum(weights)

    T = length(demands)

    Problem = Model(optimizer)

    C = [-1; 1]
    d = [0, D]
    a = [Cu,-Co]
    b(order) = [-Cu*order, Co*order]

    @variables(Problem, begin
                            D >= order >= 0
                                 λ >= 0
                                 γ[t=1:T]
                                 z[t=1:T,i=1:2,j=1:2] >= 0
                                 w[t=1:T,i=1:2]
                        end)

    for t in 1:T
        for i in 1:2
            @constraints(Problem, begin
                                        # b(order)[i] + w[t,i]*demands[t] + (1/4)*(1/λ)*w[t,i]^2 + z[t,i,:]'*d <= γ[t] 
                                        # <==> w[t,i]^2 <= 2*(2*λ)*(γ[t] - b(order)[i] - w[t,i]*demands[t] - z[t,i,:]'*d) 
                                        # <==>
                                        [2*λ; γ[t] - b(order)[i] - w[t,i]*demands[t] - z[t,i,:]'*d; w[t,i]] in MathOptInterface.RotatedSecondOrderCone(3)
                                        
                                        a[i] - C'*z[t,i,:] == w[t,i]
                                  end)
        end
    end

    @objective(Problem, Min, ε*λ + weights'*γ)

    optimize!(Problem)

    try; return value(order); catch; return W1_newsvendor_order(ε, demands, weights); end

end


ε = [LinRange(1,10,10); LinRange(20,100,9); LinRange(200,1000,9); LinRange(2000,10000,9)]
s = round.(Int, LinRange(1,history_length,34))
α = [LinRange(0.0001,0.001,10); LinRange(0.002,0.01,9); LinRange(0.02,0.1,9); LinRange(0.2,1.0,9)]
ϱ = [[0]; LinRange(0.1,1,10); LinRange(2,10,9); LinRange(20,100,9); LinRange(200,1000,9)]

train_and_test(W2_newsvendor_order, ε, windowing_weights, [history_length])
train_and_test(W2_newsvendor_order, ε, windowing_weights, s)
train_and_test(W2_newsvendor_order, ε, smoothing_weights, α)
train_and_test(W2_newsvendor_order, ε, W2_weights, ϱ)


function REMK_intersection_based_W2_newsvendor_order(_, demands, ball_radii)

    K = length(ball_radii)
    demands = demands[end-K+1:end]


    Problem = Model(optimizer)

    C = [-1; 1]
    d = [0, D]
    a = [Cu,-Co]
    b(order) = [-Cu*order, Co*order]

    @variables(Problem, begin
                            D >= order >= 0
                                 λ[k=1:K] >= 0
                                 γ[k=1:K]
                                 z[i=1:2,j=1:2] >= 0
                                 w[i=1:2,k=1:K]
                                 s[i=1:2,k=1:K]
                        end)

    for i in 1:2
        @constraints(Problem, begin
                                    # b(order)[i] + sum(w[i,k]*demands[k] + (1/4)*(1/λ[k])*w[i,k]^2 for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K)
                                    # <==> b(order)[i] + sum(w[i,k]*demands[k] + s[i,k] for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K)
                                    # (1/4)*(1/λ[K])*w[i,k]^2 <= s[i,k] for all i,k,
                                    # <==> b(order)[i] + sum(w[i,k]*demands[k] + s[i,k] for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K)
                                    # [2*λ[k]; s[i,k]; w[i,k]] in MathOptInterface.RotatedSecondOrderCone(3) for all i,k,
                                    # <==>
                                    b(order)[i] + sum(w[i,k]*demands[k] + s[i,k] for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K)
                                    a[i] - C'*z[i,:] == sum(w[i,k] for k in 1:K)
                                end)

        for k in 1:K
            @constraints(Problem, begin
                                        [2*λ[k]; s[i,k]; w[i,k]] in MathOptInterface.RotatedSecondOrderCone(3) 
                                    end)
        end
    end

    @objective(Problem, Min, sum(ball_radii[k]*λ[k] for k in 1:K) + sum(γ[k] for k in 1:K))

    optimize!(Problem)

    try; return value(order); catch; return REMK_intersection_based_W2_newsvendor_order(2*ε, demands, 2*ϱ); end

end

REMK_intersection_ball_radii(K, ε, ϱ) = [ε+(K-k)*ϱ for k in K:-1:1]

ε = [LinRange(100,1000,10); LinRange(2000,10000,9); LinRange(20000,100000,9); LinRange(200000,1000000,9)]
ϱ = [LinRange(1,10,10); LinRange(20,100,9); LinRange(200,1000,9); LinRange(2000,10000,9)]

train_and_test(REMK_intersection_based_W2_newsvendor_order, ε, REMK_intersection_ball_radii, ϱ)

#close(results_file)