using  Random, Statistics, StatsBase, Distributions

Random.seed!(42)

shift_distribution = Uniform(-0.0005,0.0005)

number_of_consumers = 10000
initial_demand_probability = 0.1

repetitions = 100
history_length = 100

windowing_parameters = round.(Int, vcat(LinRange(1,51,51), history_length))
SES_parameters = LinRange(0.0001,1.0,51)

using IterTools
ρ_ϵ_parameters = vec(collect(IterTools.product(LinRange(0.0,0.1,11), LinRange(0.0,1.0,11))))

function generate_demand_sequences(T)
    demand_sequences = [zeros(T+1) for _ in 1:repetitions]

    for repetition in 1:repetitions
        demand_probability = initial_demand_probability
        for t in 1:T+1
            demand_sequences[repetition][t] = rand(Binomial(number_of_consumers, demand_probability))
            demand_probability = min(max(demand_probability + rand(shift_distribution), 0), 1.0)
        end
    end

    return demand_sequences
end

demand_sequences = generate_demand_sequences(history_length)

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

plt = plot(1:history_length, 
        demand_sequences[1][1:end-1], 
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

using ProgressBars, IterTools
function train(parameters, solve_for_weights)

    parameter_costs_per_repetition = zeros((length(parameters), repetitions))
    Threads.@threads for parameter_index in ProgressBar(1:length(parameters))
        for repetition in 1:repetitions  
            local samples = demand_sequences[repetition][1:history_length]
            local sample_weights = solve_for_weights(samples, parameters[parameter_index])
            local order_quantity = newsvendor_order_quantity(samples; weights = sample_weights)
            parameter_costs_per_repetition[parameter_index, repetition] = newsvendor_loss(order_quantity, demand_sequences[repetition][history_length+1])
        end
    end

    return parameter_costs_per_repetition
end

windowing_costs = train(windowing_parameters, windowing_weights)
windowing_parameter_index = argmin(vec(mean(windowing_costs, dims=2)))
windowing_parameter = windowing_parameters[windowing_parameter_index]
windowing_cost = minimum(vec(mean(windowing_costs, dims=2)))
SAA_cost = mean(windowing_costs[end, :])
display("SAA cost: $SAA_cost")

display("Optimal windowing cost: $windowing_cost parameter: $windowing_parameter")

SES_costs = train(SES_parameters, SES_weights)
SES_parameter_index = argmin(vec(mean(SES_costs, dims=2)))
SES_parameter = SES_parameters[SES_parameter_index]
SES_cost = minimum(vec(mean(SES_costs, dims=2)))
display("Optimal SES cost: $SES_cost parameter: $SES_parameter")

μ = mean(SES_costs[SES_parameter_index,:] - windowing_costs[windowing_parameter_index,:])
s = sem(SES_costs[SES_parameter_index,:] - windowing_costs[windowing_parameter_index,:])
display("SES - windowing: $μ ± $s")

ρ_ϵ_costs = train(ρ_ϵ_parameters, optimal_weights)
ρ_ϵ_parameter_index = argmin(vec(mean(ρ_ϵ_costs, dims=2)))
ρ_ϵ_parameter = ρ_ϵ_parameters[ρ_ϵ_parameter_index]
ρ_ϵ_cost = minimum(vec(mean(ρ_ϵ_costs, dims=2)))
display("Optimal cost: $ρ_ϵ_cost parameter: $ρ_ϵ_parameter")