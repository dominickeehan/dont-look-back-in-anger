using  Random, Statistics, StatsBase, Distributions
using ProgressBars, IterTools

repetitions = 10
history_length = 100


initial_demand_probability = 0.1 # 0.1


include("weights.jl")

include("newsvendor-optimizations.jl")

# newsvendor_loss(order, demand) = Cu*max(demand-order,0) + Co*max(order-demand,0)

function expected_newsvendor_loss(order, demand_probability)

    expected_underage = D*demand_probability*(1-cdf(Binomial(D-1,demand_probability),order-2)) -
        order*(1-cdf(Binomial(D,demand_probability),order-1))

    expected_overage = order*cdf(Binomial(D,demand_probability),order) -
        D*demand_probability*(cdf(Binomial(D-1,demand_probability),order-1))

    return Cu*expected_underage + Co*expected_overage

end


Random.seed!(42)
U = 10^(-1) # Why W2 messed up for 10^(-1)?
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


function parameter_fit(newsvendor_value_and_order, ambiguity_radii, compute_weights, weight_parameters)

    costs = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in 1:repetitions]

    Threads.@threads for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
        weights = compute_weights(history_length, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index])

        for repetition in 1:repetitions
            demand_samples = demand_sequences[repetition][1:history_length]
            _, order = newsvendor_value_and_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
            costs[repetition][ambiguity_radius_index, weight_parameter_index] = expected_newsvendor_loss(order, demand_probability[repetition][history_length+1])

        end
    end

    ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][ambiguity_radius_index, weight_parameter_index] for repetition in 1:repetitions]

    μ = mean(minimal_costs)
    σ = sem(minimal_costs)
    display("$μ ± $σ")
    
    display([ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index]])

end

s = [round.(Int, LinRange(1,10,10)); round.(Int, LinRange(12,30,10)); round.(Int, LinRange(33,60,10)); round.(Int, LinRange(64,100,10));]
α = [[0]; LinRange(0.0001,0.001,10); LinRange(0.002,0.01,9); LinRange(0.02,0.1,9); LinRange(0.2,1.0,9)]

#ε = [[0]; LinRange(1,10,10); LinRange(20,100,9); LinRange(200,1000,9); LinRange(2000,10000,9)]
ε = 10*[[0]; LinRange(10,100,10); LinRange(200,1000,9); LinRange(2000,10000,9); LinRange(20000,100000,9)]
ϱ_divided_by_ε = [[0]; LinRange(0.0001,0.001,10); LinRange(0.002,0.01,9); LinRange(0.02,0.1,9); LinRange(0.2,1,9)]

parameter_fit(SO_newsvendor_value_and_order, ε, smoothing_weights, α)

#parameter_fit(W1_newsvendor_value_and_order, ε, windowing_weights, [history_length])
#parameter_fit(W1_newsvendor_value_and_order, ε, windowing_weights, s)
parameter_fit(W1_newsvendor_value_and_order, ε, smoothing_weights, α)
#parameter_fit(W1_newsvendor_value_and_order, ε, W1_concentration_weights, ϱ_divided_by_ε)

#parameter_fit(W2_newsvendor_value_and_order, ε, windowing_weights, [history_length])
#parameter_fit(W2_newsvendor_value_and_order, ε, windowing_weights, s)
#parameter_fit(W2_newsvendor_value_and_order, ε, smoothing_weights, α)
#parameter_fit(W2_newsvendor_value_and_order, ε, W2_concentration_weights, ϱ_divided_by_ε)

ε = [LinRange(100,1000,10); LinRange(2000,10000,9); LinRange(20000,100000,9); LinRange(200000,1000000,9)]
ϱ = [[0]; LinRange(1,10,10); LinRange(20,100,9); LinRange(200,1000,9); LinRange(2000,10000,9)]

parameter_fit(REMK_intersection_based_W2_newsvendor_value_and_order, ε, REMK_intersection_ball_radii, ϱ)