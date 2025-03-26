using  Random, Statistics, StatsBase, Distributions

Random.seed!(42)

shift_distribution = Uniform(-0.0005,0.0005)

number_of_consumers = 10000
initial_demand_probability = 0.1

repetitions = 300
history_length = 200
training_length = round(Int, 0.3*history_length)

windowing_parameters = round.(Int, vcat(LinRange(1,51,11), history_length))
SES_parameters = LinRange(0.0001,0.25,11)

ambiguity_radii = LinRange(0.0,20,11)


demand_sequences = [zeros(history_length+1) for _ in 1:repetitions]
for repetition in 1:repetitions
    demand_probability = initial_demand_probability
    for t in 1:history_length+1
        demand_sequences[repetition][t] = rand(Binomial(number_of_consumers, demand_probability))
        demand_probability = min(max(demand_probability + rand(shift_distribution), 0), 1.0)
    end
end


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

Cu = 1 # Cost of underage.
Co = 2/3 # Cost of overage.

newsvendor_loss(x,ξ) = Cu*max(ξ-x,0) + Co*max(x-ξ,0)

using JuMP
using Gurobi
env = Gurobi.Env() 
optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env), "OutputFlag" => 0)#, "Method" => 1,)

#using COPT
#optimizer = optimizer_with_attributes(COPT.Optimizer, "Logging" => 0, "LogToConsole" => 0)  

D = number_of_consumers

function newsvendor_order(ε, ξ, weights) 

    T = length(ξ)

    Problem = Model(optimizer)

    C = [-1; 1]
    d = [0, D]
    a = [Cu,-Co]
    b(x) = [-Cu*x, Co*x]

    @variables(Problem, begin
                            D >= x >= 0
                                 λ
                                 s[t=1:T]
                                 γ[t=1:T,i=1:2,j=1:2] >= 0
                                 z[t=1:T,i=1:2] 
                        end)

    for t in 1:T
        for i in 1:2
            @constraints(Problem, begin
                                        b(x)[i] + a[i]*ξ[t] + γ[t,i,:]'*(d-C*ξ[t]) <= s[t] #b(x)[i] + a[i]*ξ[t] + sum(γ[t,i,j]*(d[j]-((C*ξ[t])[j])) for j in 1:2) <= s[t]
                                        z[t,i] <= λ
                                        C'*γ[t,i,:] - a[i] <= z[t,i]
                                       -C'*γ[t,i,:] + a[i] <= z[t,i]
                                  end)
        end
    end

    @objective(Problem, Min, ε*λ + weights'*s)

    optimize!(Problem)

    return value(x)

end
#display(newsvendor_order(0.0, [1, 2, 3], [1/3, 1/3, 1/3]))

#newsvendor_order(ε, ξ, weights) = quantile(ξ, Weights(weights), Cu/(Co+Cu)) 
#display(newsvendor_order([0.0], [1,2,3], [1/3,1/3,1/3]))

using ProgressBars, IterTools
function train_and_test(ambiguity_radii, compute_weights, weight_parameters)

    repetition_costs = zeros(repetitions)
    Threads.@threads for repetition in ProgressBar(1:repetitions)

        training_costs = [zeros((length(ambiguity_radii), length(weight_parameters))) for t in 1:training_length]

        for ambiguity_radius_index in eachindex(ambiguity_radii)
            for weight_parameter_index in eachindex(weight_parameters)
                for t in 1:training_length
                    demand_samples = demand_sequences[repetition][1:history_length-1-training_length+t]
                    weights = compute_weights(demand_samples, weight_parameters[weight_parameter_index])
                    order = newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
                    training_costs[t][ambiguity_radius_index, weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length-1-training_length+t+1])
                end
            end
        end

        ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(training_costs)))
        demand_samples = demand_sequences[repetition][1:history_length]
        weights = compute_weights(demand_samples, weight_parameters[weight_parameter_index])
        order = newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
        repetition_costs[repetition] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])
    end

    return repetition_costs
end

include("weights.jl")

SAA_costs = train_and_test(ambiguity_radii, windowing_weights, [history_length])
μ = mean(SAA_costs)
s = sem(SAA_costs)
display("SAA cost: $μ ± $s")

windowing_costs = train_and_test(ambiguity_radii, windowing_weights, windowing_parameters)
μ = mean(windowing_costs)
s = sem(windowing_costs)
display("Windowing cost: $μ ± $s")

SES_costs = train_and_test(ambiguity_radii, SES_weights, SES_parameters)
μ = mean(SES_costs)
s = sem(SES_costs)
display("SES cost: $μ ± $s")
