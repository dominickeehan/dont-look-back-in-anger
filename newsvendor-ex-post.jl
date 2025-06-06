using Random, Statistics, StatsBase, Distributions
using ProgressBars, IterTools

repetitions = 100
history_length = 30


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
u = 0.001
shift_distribution = Uniform(-u,u)

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
            local _, order, doubling_count[repetition][ambiguity_radius_index, weight_parameter_index] = 
                newsvendor_value_and_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights, 0)
            costs[repetition][ambiguity_radius_index, weight_parameter_index] = expected_newsvendor_loss(order, demand_probability[repetition][history_length+1])

        end
    end

    ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][ambiguity_radius_index, weight_parameter_index] for repetition in 1:repetitions]

    μ = mean(minimal_costs)
    σ = sem(minimal_costs)
    display("$μ ± $σ")
    
    display([ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index]])
    display(mean([doubling_count[repetition][ambiguity_radius_index, weight_parameter_index] for repetition in 1:repetitions]))

end

ε = [[0]; LinRange(1e0,1e1,10); LinRange(2e1,1e2,9); LinRange(2e2,1e3,9); LinRange(2e3,1e4,9); LinRange(2e4,1e5,9)]
s = [round.(Int, LinRange(1,10,10)); round.(Int, LinRange(12,30,10)); round.(Int, LinRange(33,60,10)); round.(Int, LinRange(64,100,10))]

GeomRange(a, b, n) = exp.(LinRange(log(a), log(b), n))

α = [[0]; GeomRange(1e-4,1e0,40)]
ϱ╱ε = [[0]; GeomRange(1e-4,1e0,40)]

#parameter_fit(SO_newsvendor_value_and_order, ε, smoothing_weights, α)

#parameter_fit(W1_newsvendor_value_and_order, ε, windowing_weights, [history_length])
#parameter_fit(W1_newsvendor_value_and_order, ε, windowing_weights, s)
#parameter_fit(W1_newsvendor_value_and_order, ε, smoothing_weights, α)
#parameter_fit(W1_newsvendor_value_and_order, ε, W1_concentration_weights, ϱ╱ε)

#parameter_fit(W1_newsvendor_value_and_order, 0, windowing_weights, s)
parameter_fit(W1_newsvendor_value_and_order, 0, smoothing_weights, α)
#parameter_fit(W1_newsvendor_value_and_order, 0, W1_concentration_weights, ϱ╱ε)

#parameter_fit(W2_newsvendor_value_and_order, ε, windowing_weights, [history_length])
#parameter_fit(W2_newsvendor_value_and_order, ε, windowing_weights, s)
#parameter_fit(W2_newsvendor_value_and_order, ε, smoothing_weights, α)
#parameter_fit(W2_newsvendor_value_and_order, ε, W2_concentration_weights, ϱ_divided_by_ε)

ε = [LinRange(1e2,1e3,10); LinRange(2e3,1e4,9); LinRange(2e4,1e5,9); LinRange(2e5,1e6,9); LinRange(2e6,1e7,9)]
ϱ╱ε = [[0]; GeomRange(1e-2,1e2,40)]

parameter_fit(REMK_intersection_W2_newsvendor_value_and_order, ε, REMK_intersection_weights, ϱ╱ε)