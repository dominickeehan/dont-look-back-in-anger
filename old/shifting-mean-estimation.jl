using Random, Statistics, StatsBase, Distributions

using LinearAlgebra

Random.seed!(42)

m = 10

shift_distribution = MvNormal(zeros(m), ones(m)) # Product(fill(Uniform(-2,2), m))

initial_mean = zeros(m)

repetitions = 2000
history_length = 1000

windowing_parameters = round.(Int, vcat(LinRange(1,51,51), history_length))
SES_parameters = LinRange(0.0001,1.0,51)

function generate_sample_sequences(T)
    sample_sequences = [[zeros(m) for _ in 1:T+1] for _ in 1:repetitions]

    for repetition in 1:repetitions
        mean = initial_mean
        for t in 1:T+1
            sample_sequences[repetition][t] = rand(MvNormal(mean, ones(m)))
            mean = 0.01*initial_mean + 0.99*mean + rand(shift_distribution)
        end
    end

    return sample_sequences
end

sample_sequences = generate_sample_sequences(history_length)

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
        stack(sample_sequences[1][1:end-1])', 
        xlabel = "Time (units)", 
        ylabel = "sample (units)",
        labels = nothing, 
        #color = palette(:tab10)[1],
        #markershape = :circle,
        #linewidth = 0,
        #markersize = 3.5, 
        #markerstrokewidth = 0.875,
        #markerstrokecolor = :black,
        topmargin = 0pt, 
        rightmargin = 0pt,
        bottommargin = 5pt, 
        leftmargin = 5pt,
        )

display(plt)

Co = 1 # Cost of overage.
Cu = 1/3 # Cost of underage.

loss(x,ξ) = norm(x-ξ, 2)^2
function sample_mean(samples; weights = nothing) 
    if weights === nothing
        return mean(samples)
    else
        return mean(samples, Weights(weights))
    end
end

include("weights.jl")

using ProgressBars, IterTools
function train(parameters, solve_for_weights)

    parameter_costs_per_repetition = zeros((length(parameters), repetitions))
    Threads.@threads for parameter_index in ProgressBar(1:length(parameters))
        for repetition in 1:repetitions  
            local samples = sample_sequences[repetition][1:history_length]
            local sample_weights = solve_for_weights(samples, parameters[parameter_index])
            local mean_estimate = sample_mean(samples; weights = sample_weights)
            parameter_costs_per_repetition[parameter_index, repetition] = loss(mean_estimate, sample_sequences[repetition][history_length+1])
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