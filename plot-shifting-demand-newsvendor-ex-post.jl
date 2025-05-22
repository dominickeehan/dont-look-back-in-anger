using  Random, Statistics, StatsBase, Distributions
using ProgressBars, IterTools
using Plots, Measures

number_of_consumers = 1e4
D = number_of_consumers

Cu = 4 # Cost of underage.
Co = 1 # Cost of overage.

newsvendor_loss(x,ξ) = Cu*max(ξ-x,0) + Co*max(x-ξ,0)

newsvendor_order(ε, ξ, weights) = quantile(ξ, Weights(weights), Cu/(Co+Cu))

W1_newsvendor_order(ε, ξ, weights) = newsvendor_order(ε, ξ, weights)

using JuMP, MathOptInterface
using Gurobi
env = Gurobi.Env() 
GRBsetintparam(env, "OutputFlag", 0)
GRBsetintparam(env, "BarHomogeneous", 1)
optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

function W2_newsvendor_order(ε, ξ, weights) 

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
                                        # b(x)[i] + w[t,i]*ξ[t] + (1/4)*(1/λ)*w[t,i]^2 + z[t,i,:]'*d <= γ[t] <==>
                                        # w[t,i]^2 <= 2*(2*λ)*(γ[t] - b(x)[i] - w[t,i]*ξ[t] - z[t,i,:]'*d) <==>
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


function REMK_intersection_based_W₂_newsvendor_order(ball_radii, ξ, empty_counter) 

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
                                    # <==> b(x)[i] + sum(w[i,k]*ξ[k] + s[i,k] for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K),      (1/4)*(1/λ[K])*w[i,k]^2 <= s[i,k] for all i,k
                                    # <==> b(x)[i] + sum(w[i,k]*ξ[k] + s[i,k] for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K),      [2*λ[k]; s[i,k]; w[i,k]] in MathOptInterface.RotatedSecondOrderCone(3) for all i,k
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
        return value(x), empty_counter
    catch
        return REMK_intersection_based_W₂_newsvendor_order(2*ball_radii,ξ,1)
    end
end

include("weights.jl")

Random.seed!(42)

repetitions = 200
history_length = 100

digits = 4

function ex_post_costs(demand_sequences, Wp_newsvendor_order, ambiguity_radii, compute_weights, weight_parameters)

    costs = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in 1:repetitions]
    for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))

        weights = compute_weights(history_length, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index])
        Threads.@threads for repetition in 1:repetitions

            demand_samples = demand_sequences[repetition][1:history_length]
            order = Wp_newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
            
            costs[repetition][ambiguity_radius_index, weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])
        end
    end

    ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][ambiguity_radius_index, weight_parameter_index] for repetition in 1:repetitions]

    return round(mean(minimal_costs), digits=digits), round(sem(minimal_costs), digits=digits)
end

K = history_length
function intersection_ex_post_costs(demand_sequences, initial_ball_radii_parameters, shift_bound_parameters)

    costs = [zeros((length(initial_ball_radii_parameters),length(shift_bound_parameters))) for repetition in 1:repetitions]
    empty_counters = [zeros((length(initial_ball_radii_parameters),length(shift_bound_parameters))) for repetition in 1:repetitions]
    for (initial_ball_radius_index, shift_bound_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(initial_ball_radii_parameters), eachindex(shift_bound_parameters))))

        ball_radii = reverse([initial_ball_radii_parameters[initial_ball_radius_index]+(k-1)*shift_bound_parameters[shift_bound_parameter_index] for k in 1:K])
        Threads.@threads for repetition in 1:repetitions

            demand_samples = demand_sequences[repetition][1:history_length]
            order, empty_counter = REMK_intersection_based_W₂_newsvendor_order(ball_radii, demand_samples, 0)
            
            costs[repetition][initial_ball_radius_index, shift_bound_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])
            empty_counters[repetition][initial_ball_radius_index, shift_bound_parameter_index] = empty_counter
        
        end
    end

    initial_ball_radius_index, shift_bound_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][initial_ball_radius_index, shift_bound_parameter_index] for repetition in 1:repetitions]
    empty_frequency = mean([empty_counters[repetition][initial_ball_radius_index, shift_bound_parameter_index] for repetition in 1:repetitions])

    return round(mean(minimal_costs), digits=digits), round(sem(minimal_costs), digits=digits)
end

uniform_shift_sizes = [0.0001,0.001,0.005,0.01]

windowing_parameters = round.(Int, LinRange(1,history_length,30))

W1_windowing_costs = zeros(length(uniform_shift_sizes))
W1_windowing_sems = zeros(length(uniform_shift_sizes))

ambiguity_radii = [LinRange(1,10,10); LinRange(20,100,9); LinRange(200,1000,9); LinRange(2000,10000,9)]

W2_windowing_costs = zeros(length(uniform_shift_sizes))
W2_windowing_sems = zeros(length(uniform_shift_sizes))

initial_ball_radii_parameters = [LinRange(100,1000,10); LinRange(2000,10000,9); LinRange(20000,100000,9)]
shift_bound_parameters = [LinRange(100,1000,10); LinRange(2000,10000,9); LinRange(20000,100000,9)]

intersection_costs = zeros(length(uniform_shift_sizes))
intersection_sems = zeros(length(uniform_shift_sizes))

for uniform_shift_size_index in eachindex(uniform_shift_sizes) 

    initial_demand_probability = 0.1
    shift_distribution = Uniform(-uniform_shift_sizes[uniform_shift_size_index],uniform_shift_sizes[uniform_shift_size_index])

    demand_sequences = [zeros(history_length+1) for _ in 1:repetitions]
    for repetition in 1:repetitions

        demand_probability = initial_demand_probability
        for t in 1:history_length+1

            demand_sequences[repetition][t] = rand(Binomial(number_of_consumers, demand_probability))
            demand_probability = min(max(demand_probability + rand(shift_distribution), 0), 1.0)
        end
    end

    W1_windowing_costs[uniform_shift_size_index], W1_windowing_sems[uniform_shift_size_index] = 
        ex_post_costs(demand_sequences, W1_newsvendor_order, [0], windowing_weights, windowing_parameters)
    
    W2_windowing_costs[uniform_shift_size_index], W2_windowing_sems[uniform_shift_size_index] = 
        ex_post_costs(demand_sequences, W2_newsvendor_order, ambiguity_radii, windowing_weights, windowing_parameters)

    intersection_costs[uniform_shift_size_index], intersection_sems[uniform_shift_size_index] = 
        intersection_ex_post_costs(demand_sequences, initial_ball_radii_parameters, shift_bound_parameters)

end

plot(uniform_shift_sizes, W1_windowing_costs, xscale=:log10)#, ribbon=W1_windowing_sems)
plot!(uniform_shift_sizes, W2_windowing_costs, xscale=:log10)#, ribbon=W2_windowing_sems)
plot!(uniform_shift_sizes, intersection_costs, xscale=:log10)#, ribbon=intersection_sems)




