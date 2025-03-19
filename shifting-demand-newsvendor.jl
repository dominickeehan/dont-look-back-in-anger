using  Random, Statistics, StatsBase, Distributions

Random.seed!(42)

shift_distribution = Uniform(-0.005,0.005)

number_of_consumers = 1000
initial_demand_probability = 0.1

repetitions = 500
history_length = 1000

plot_history_length = 1000

windowing_parameters = round.(Int, LinRange(1,T,51))
SES_parameters = LinRange(0.001,1.0,51)

function generate_shifted_demand_probabilities(T)
    demand_probabilities = zeros(T+1)
    demand_probabilities[1] = initial_demand_probability

    for t in 2:T+1
        demand_probabilities[t] = min(max(demand_probabilities[t-1] + rand(shift_distribution), 0), 1)
    end

    return demand_probabilities
end

function generate_demand_sequences_per_repetition(shifted_demand_probabilities)
    T = length(shifted_demand_probabilities)
    demands = [zeros(T) for _ in 1:repetitions]

    for repetition in 1:repetitions
        for t in 1:T
            demands[repetition][t] = rand(Binomial(number_of_consumers,shifted_demand_probabilities[t]))
        end
    end

    return demands
end

demands = generate_demand_sequences_per_repetition(generate_shifted_demand_probabilities(plot_history_length-1))

using Plots, Measures

default() # Reset plot defaults.

gr(size = (600,400))

font_family = "Computer Modern"
primary_font = Plots.font(font_family, pointsize = 17)
secondary_font = Plots.font(font_family, pointsize = 11)
legend_font = Plots.font(font_family, pointsize = 16)

default(framestyle = :box,
        grid = true,
        #gridlinewidth = 1.0,
        gridalpha = 0.075,
        #minorgrid = true,
        #minorgridlinewidth = 1.0, 
        #minorgridalpha = 0.075,
        #minorgridlinestyle = :dash,
        tick_direction = :in,
        xminorticks = 0, 
        yminorticks = 0,
        fontfamily = font_family,
        guidefont = primary_font,
        tickfont = secondary_font,
        legendfont = legend_font)

plt = plot(1:plot_history_length, 
        demands, 
        xlabel = "Time (units)", 
        ylabel = "Demand (units)",
        labels = nothing, 
        color = palette(:tab10)[1],
        markershape = :circle,
        linewidth = 0,
        markersize = 3.5, 
        markerstrokewidth = 0.875,
        markerstrokecolor = :black,
        topmargin = 0pt, 
        rightmargin = 0pt,
        bottommargin = 5pt, 
        leftmargin = 5pt,
        )

display(plt)

Co = 1 # Cost of overage.
Cu = 1/3 # Cost of underage.

newsvendor_loss(x,d) = Co*max(x-d,0) + Cu*max(d-x,0)
function newsvendor_order_quantity(demands; weights = nothing) 
    if weights === nothing
        return quantile(demands, Cu/(Co+Cu))
    else
        return quantile(demands, Weights(weights), Cu/(Co+Cu))
    end
end

function optimal_newsvendor_order_quantity(distribution) 
    return quantile(distribution, Cu/(Co+Cu))
end

include("weights.jl")

demand_sequences_per_repetition = generate_demand_sequences_per_repetition(generate_shifted_demand_probabilities(history_length))

using ProgressBars, IterTools
function train(parameters, solve_for_weights)

    parameter_costs = zeros((length(parameters),repetitions))
    Threads.@threads for par in ProgressBar(1:length(parameters))
        for repetition in 1:repetitions  
            local samples = demand_sequences_per_repetition[repetition][1:history_length]
            local sample_weights = solve_for_weights(samples, parameters[par])
            local order = newsvendor_order_quantity(samples; weights = sample_weights)
            parameter_costs[par,repetition] = newsvendor_loss(order, demand_sequences_per_repetition[repetition][history_length+1])
        end
    end
    parameter_costs = mean(parameter_costs, dims=2)

    return min(parameter_costs...), parameters[argmin(parameter_costs)]

end

# Windowing.

display(train(windowing_parameters, windowing_weights))

display(train(SES_parameters, SES_weights))
