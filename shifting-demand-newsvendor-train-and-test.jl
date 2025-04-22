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

function W₂_newsvendor_order(ε, ξ, weights) 

    ξ = ξ[weights .>= 1e-3]
    weights = weights[weights .>= 1e-3]
    weights .= weights/sum(weights)

    T = length(ξ)

    Problem = Model(optimizer)

    C = [-1; 1]
    d = [0, D]
    a = [Cu,-Co]
    b(x) = [-Cu*x, Co*x]

    @variables(Problem, begin
                            D >= x >= 0
                                 λ >= 0
                                 γ[t=1:T]
                                 z[t=1:T,i=1:2,j=1:2] >= 0
                                 w[t=1:T,i=1:2]
                        end)

    for t in 1:T
        for i in 1:2
            @constraints(Problem, begin
                                        # b(x)[i] + w[t,i]*ξ[t] + (1/4)*(1/λ)*w[t,i]^2 + z[t,i,:]'*d <= γ[t] <==> w[t,i]^2 <= 2*(2*λ)*(γ[t] - b(x)[i] - w[t,i]*ξ[t] - z[t,i,:]'*d) <==>
                                        [2*λ; γ[t] - b(x)[i] - w[t,i]*ξ[t] - z[t,i,:]'*d; w[t,i]] in MathOptInterface.RotatedSecondOrderCone(3)
                                        a[i] - C'*z[t,i,:] == w[t,i]
                                  end)
        end
    end

    @objective(Problem, Min, ε*λ + weights'*γ)

    optimize!(Problem)

    try
        return value(x)
    catch
        #display("catch")
        return newsvendor_order(ε, ξ, weights)
    end
end

include("weights.jl")

job_number = parse(Int64, ENV["PBS_ARRAY_INDEX"])

Random.seed!(job_number)

open("$job_number.csv", "w") do file; end

results_file = open("$job_number.csv", "a")


shift_distribution = Uniform(-0.0005,0.0005)

initial_demand_probability = 0.1

repetitions = 100
history_length = 100
training_length = 70

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

    precomputed_weights = stack([[[zeros(t-1) for t in 71:100] for ambiguity_radius_index in eachindex(ambiguity_radii)] for weight_parameter_index in eachindex(weight_parameters)])

    println("precomputing_weights...")

    for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
    #Threads.@threads for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
        for t in 71:100
            precomputed_weights[ambiguity_radius_index, weight_parameter_index][t-70] = compute_weights(t-1, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index])
        end
    end

    println("training and testing W₂ method...")

    for repetition in ProgressBar(1:repetitions)
        
        training_costs = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in 71:100]
        for ambiguity_radius_index in eachindex(ambiguity_radii)
            for weight_parameter_index in eachindex(weight_parameters)
                
                for t in 71:100
                
                    weights = precomputed_weights[ambiguity_radius_index, weight_parameter_index][t-70]

                    demand_samples = demand_sequences[repetition][1:t-1]
                    order = W₂_newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
                    training_costs[t-70][ambiguity_radius_index, weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][t])
                end
            end
        end

        ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(training_costs)))
        weights = compute_weights(100, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index])
        demand_samples = demand_sequences[repetition][1:100]
        order = W₂_newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)

        cost = newsvendor_loss(order, demand_sequences[repetition][101])
        ambiguity_radius_to_test = ambiguity_radii[ambiguity_radius_index]
        weight_parameter_to_test = weight_parameters[weight_parameter_index]

        println(results_file, "$cost, $ambiguity_radius_to_test, $weight_parameter_to_test")

    end
end

ambiguity_radii = [LinRange(1,10,4); LinRange(40,100,3)]
shift_bound_parameters = [LinRange(0.1,1,4); LinRange(4,10,3)]
train_and_test(ambiguity_radii, W₂_concentration_weights, shift_bound_parameters)

windowing_parameters = round.(Int, LinRange(10,history_length,7))
train_and_test(ambiguity_radii, windowing_weights, windowing_parameters)

smoothing_parameters = LinRange(0.02,0.2,7)
train_and_test(ambiguity_radii, smoothing_weights, smoothing_parameters)



function REMK_intersection_based_W₂_newsvendor_order(ball_radii, ξ) 

    K = length(ball_radii)
    ξ = ξ[end-K+1:end]

    Problem = Model(optimizer)

    C = [-1; 1]
    d = [0, D]
    a = [Cu,-Co]
    b(x) = [-Cu*x, Co*x]

    @variables(Problem, begin
                            D >= x >= 0
                                 λ[k=1:K] >= 0
                                 γ[k=1:K]
                                 z[i=1:2,j=1:2] >= 0
                                 w[i=1:2,k=1:K]
                                 s[i=1:2,k=1:K]
                        end)

    for i in 1:2
        @constraints(Problem, begin
                                    # b(x)[i] + sum(w[i,k]*ξ[k] + (1/4)*(1/λ[k])*w[i,k]^2 for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K)
                                    # ⟺ b(x)[i] + sum(w[i,k]*ξ[k] + s[i,k] for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K),      (1/4)*(1/λ[K])*w[i,k]^2 <= s[i,k]
                                    # ⟺ b(x)[i] + sum(w[i,k]*ξ[k] + s[i,k] for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K),      [2*λ[k]; s[i,k]; w[i,k]] in MathOptInterface.RotatedSecondOrderCone(3) ∀i,k
                                    b(x)[i] + sum(w[i,k]*ξ[k] + s[i,k] for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K)
                                    a[i] - C'*z[i,:] == sum(w[i,k] for k in 1:K)
                                end)
    end

    for i in 1:2
        for k in 1:K
            @constraints(Problem, begin
                                        [2*λ[k]; s[i,k]; w[i,k]] in MathOptInterface.RotatedSecondOrderCone(3) 
                                    end)
        end
    end

    @objective(Problem, Min, sum(ball_radii[k]*λ[k] for k in 1:K) + sum(γ[k] for k in 1:K))

    optimize!(Problem)

    try
        return value(x)
    catch
        #display("catch")
        return REMK_intersection_based_W₂_newsvendor_order(2*ball_radii,ξ)
    end
end


function train_and_test(initial_ball_radii_parameters, shift_bound_parameters)

    costs = zeros(repetitions)

    println("training and testing intersection-based W₂ method...")

    for repetition in ProgressBar(1:repetitions)
        
        training_costs = [zeros((length(initial_ball_radii_parameters),length(shift_bound_parameters))) for _ in 71:100]
        for initial_ball_radius_index in eachindex(initial_ball_radii_parameters)
            for shift_bound_parameter_index in eachindex(shift_bound_parameters)
                
                for t in 71:100
                    K = t-1
                    ball_radii = reverse([initial_ball_radii_parameters[initial_ball_radius_index]+(k-1)*shift_bound_parameters[shift_bound_parameter_index] for k in 1:K])

                    demand_samples = demand_sequences[repetition][1:t-1]
                    order = REMK_intersection_based_W₂_newsvendor_order(ball_radii, demand_samples)
                    training_costs[t-70][initial_ball_radius_index, shift_bound_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][t])
                end
            end
        end

        initial_ball_radius_index, shift_bound_parameter_index = Tuple(argmin(mean(training_costs)))
        K = 100
        ball_radii = reverse([initial_ball_radii_parameters[initial_ball_radius_index]+(k-1)*shift_bound_parameters[shift_bound_parameter_index] for k in 1:K])
        demand_samples = demand_sequences[repetition][1:100]
        order = REMK_intersection_based_W₂_newsvendor_order(ball_radii, demand_samples)
        
        cost = newsvendor_loss(order, demand_sequences[repetition][101])
        ε = initial_ball_radii_parameters[initial_ball_radius_index]
        ϱ = shift_bound_parameters[shift_bound_parameter_index]

        println(results_file, "$cost, $ε, $ϱ")

    end
end

initial_ball_radii_parameters = [LinRange(100,1000,4); LinRange(4000,10000,3)]
shift_bound_parameters = [LinRange(10,100,4); LinRange(400,1000,3)]

train_and_test(initial_ball_radii_parameters, shift_bound_parameters)



close(results_file)