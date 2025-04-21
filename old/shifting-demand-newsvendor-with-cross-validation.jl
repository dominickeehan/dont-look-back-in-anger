using  Random, Statistics, StatsBase, Distributions
include("weights.jl")

Random.seed!(42)

shift_distribution = Uniform(-0.001,0.001)

number_of_consumers = 10000
initial_demand_probability = 0.1

repetitions = 1000
history_length = 60
training_length = round(Int, 0.3*history_length)

windowing_parameters = round.(Int, LinRange(1,history_length,6))
#smoothing_parameters = LinRange(0.0001,0.3,51)

ambiguity_radii = [10.0]#[0,1e0,1e1,1e2,1e3,1e4,1e5]
shift_bound_parameters = [0.075, 0.1, 0.125]#LinRange(0.01,0.2,6) #[0.1] #[0,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3]

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

using JuMP, MathOptInterface
using Gurobi
env = Gurobi.Env() 
GRBsetintparam(env, "OutputFlag", 0)
GRBsetintparam(env, "BarHomogeneous", 1)
optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

#using COPT
#optimizer = optimizer_with_attributes(COPT.ConeOptimizer, "Logging" => 0, "LogToConsole" => 0, "BarHomogeneous" => 1,)  

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
newsvendor_order(ε, ξ, weights) = quantile(ξ, Weights(weights), Cu/(Co+Cu))


display(newsvendor_order(0.0, demand_sequences[1][1:end-1], windowing_weights(history_length, 0.0, history_length)))

function W2_newsvendor_order(ε, ξ, weights) 

    ξ = ξ[weights .>= 1e-3]
    weights = weights[weights .>= 1e-3]
    weights .= weights/sum(weights)

    T = length(ξ)

    Problem = Model(optimizer)
    #set_silent(Problem)

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
                                        #b(x)[i] + w[t,i]*ξ[t] + (1/4)*(1/λ)*w[t,i]^2 + z[t,i,:]'*d <= γ[t] #<==>
                                        # w[t,i]^2 <= 2*(2*λ)*(γ[t] - b(x)[i] - w[t,i]*ξ[t] - z[t,i,:]'*d) #<==>
                                        [2*λ; γ[t] - b(x)[i] - w[t,i]*ξ[t] - z[t,i,:]'*d; w[t,i]] in MathOptInterface.RotatedSecondOrderCone(3)
                                        a[i] - C'*z[t,i,:] == w[t,i]
                                  end)
        end
    end

    @objective(Problem, Min, ε*λ + weights'*γ)

    optimize!(Problem)

    #termination_status(Problem)
    #primal_status(Problem)
    #primal_feasibility_report(Problem)
    #is_solved_and_feasible(Problem)

    #print(Problem)

    try
        return value(x)
    catch
        display("catch")
        return newsvendor_order(ε, ξ, weights)
    end
end

using ProgressBars, IterTools
function train_and_test(ambiguity_radii, compute_weights, weight_parameters)

    repetition_costs = zeros(repetitions)

    training_costs = [[zeros((length(ambiguity_radii), length(weight_parameters))) for _ in 1:training_length] for _ in 1:repetitions]

    Threads.@threads for (ambiguity_radius_index, weight_parameter_index, repetition) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters), 1:repetitions)))
        sequences_of_weights = 
            [compute_weights(history_length-1-training_length+t, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index]) for t in 1:training_length]
        
        for t in 1:training_length

            #display((ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index]))

            demand_samples = demand_sequences[repetition][1:history_length-1-training_length+t]
            weights = sequences_of_weights[t]
            order = W2_newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
            training_costs[repetition][t][ambiguity_radius_index, weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length-1-training_length+t+1])
        end
    end

    for repetition in 1:repetitions
        ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(training_costs[repetition])))

        #display((ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index]))

        demand_samples = demand_sequences[repetition][1:history_length]
        weights = compute_weights(history_length, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index])
        order = W2_newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
        repetition_costs[repetition] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])
    end

    return repetition_costs
end


SAA_costs = train_and_test(ambiguity_radii, windowing_weights, [history_length])
μ = mean(SAA_costs)
s = sem(SAA_costs)
display("SAA cost: $μ ± $s")


windowing_costs = train_and_test(ambiguity_radii, windowing_weights, windowing_parameters)
μ = mean(windowing_costs)
s = sem(windowing_costs)
display("Windowing cost: $μ ± $s")

#=
smoothing_costs = train_and_test(ambiguity_radii, smoothing_weights, smoothing_parameters)
μ = mean(smoothing_costs)
s = sem(smoothing_costs)
display("smoothing cost: $μ ± $s")
=#

#=
triangular_costs = train_and_test(ambiguity_radii, triangular_weights, windowing_parameters)
μ = mean(triangular_costs)
s = sem(triangular_costs)
display("triangular cost: $μ ± $s")
=#

optimal_costs = train_and_test(ambiguity_radii, optimal_weights, shift_bound_parameters)
μ = mean(optimal_costs)
s = sem(optimal_costs)
display("optimal cost: $μ ± $s")

#=
display(mean(optimal_costs - windowing_costs))
display(sem(optimal_costs - windowing_costs))
=#

#=



for ε in [0,1e1,1e3,1e5,1e7]
    for ϱ in [0,1e2,1e4]
        for repetition in [1]
            #
        end
    end
end
=#





#for ε in [0,1e2,1e3,1e4,1e5,1e6]; display(W2_newsvendor_order(ε, demand_sequences[1][1:end-1], windowing_weights(history_length, 0, history_length))); end