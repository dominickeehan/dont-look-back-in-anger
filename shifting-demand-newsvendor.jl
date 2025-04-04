using  Random, Statistics, StatsBase, Distributions

number_of_consumers = 10000
D = number_of_consumers

Cu = 1 # Cost of underage.
Co = 2/3 # Cost of overage.

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
                                        # b(x)[i] + w[t,i]*ξ[t] + (1/4)*(1/λ)*w[t,i]^2 + z[t,i,:]'*d <= γ[t] <==> w[t,i]^2 <= 2*(2*λ)*(γ[t] - b(x)[i] - w[t,i]*ξ[t] - z[t,i,:]'*d) <==>
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

include("weights.jl")

Random.seed!(42)

shift_distribution = Uniform(-0.002,0.002)

initial_demand_probability = 0.1

repetitions = 10000
history_length = 30

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
        stack(demand_sequences[1:5])[1:end-1,:], 
        xlabel = "Time", 
        ylabel = "Demand",
        labels = nothing, 
        #linecolor = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5]],
        #markercolor = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5]],
        #markershape = :auto,
        color = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5]], #palette(:tab10)[1],
        linewidth = 1,
        #alpha = 1,
        #linestyle = :auto,
        #markersize = 4, 
        #markerstrokewidth = 1,
        #markerstrokecolor = :black,
        topmargin = 0pt, 
        rightmargin = 0pt,
        bottommargin = 5pt, 
        leftmargin = 5pt,
        )

display(plt)

#savefig(plt, "figures/demand_sequences.pdf")


using ProgressBars, IterTools
function train(ambiguity_radii, compute_weights, weight_parameters)

    costs = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in 1:repetitions]

    Threads.@threads for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
        weights = compute_weights(history_length, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index])

        for repetition in 1:repetitions
            demand_samples = demand_sequences[repetition][1:history_length]
            order = W1_newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
            costs[repetition][ambiguity_radius_index, weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])

        end
    end

    ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][ambiguity_radius_index, weight_parameter_index] for repetition in 1:repetitions]
    display((ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index]))

    return minimal_costs
end

ambiguity_radii = LinRange(10,100,history_length) # Only matters for optimal costs.
windowing_parameters = round.(Int, LinRange(3,history_length,history_length))
smoothing_parameters = LinRange(0.0001,0.4,history_length)
shift_bound_parameters = LinRange(1,10,history_length)


W1_naive_costs = train([0], windowing_weights, [history_length])
μ = mean(W1_naive_costs)
s = sem(W1_naive_costs)
display("W1 naive cost: $μ ± $s")

W1_windowing_costs = train([0], windowing_weights, windowing_parameters)
μ = mean(W1_windowing_costs)
s = sem(W1_windowing_costs)
display("W1 windowing cost: $μ ± $s")

W1_smoothing_costs = train([0], smoothing_weights, smoothing_parameters)
μ = mean(W1_smoothing_costs)
s = sem(W1_smoothing_costs)
display("W1 smoothing cost: $μ ± $s")
display(plot(1:history_length, reverse(smoothing_weights(history_length, [0], 0.27))))

W1_optimal_costs = train(ambiguity_radii, W1_optimal_weights, shift_bound_parameters)
μ = mean(W1_optimal_costs)
s = sem(W1_optimal_costs)
display("W1 optimal cost: $μ ± $s")
display(plot(1:history_length, reverse(W1_optimal_weights(history_length, 21.1, 2.6))))




function train(ambiguity_radii, compute_weights, weight_parameters)

    costs = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in 1:repetitions]

    Threads.@threads for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
        weights = compute_weights(history_length, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index])

        for repetition in 1:repetitions
            demand_samples = demand_sequences[repetition][1:history_length]
            order = W2_newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
            costs[repetition][ambiguity_radius_index, weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])

        end
    end

    ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][ambiguity_radius_index, weight_parameter_index] for repetition in 1:repetitions]
    display((ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index]))

    return minimal_costs
end

ambiguity_radii = LinRange(100,500,11)
windowing_parameters = round.(Int, LinRange(3,history_length,11))
smoothing_parameters = LinRange(0.0001,0.4,11)
shift_bound_parameters = LinRange(10,30,11)

W2_naive_costs = train(ambiguity_radii, windowing_weights, [history_length])
μ = mean(W2_naive_costs)
s = sem(W2_naive_costs)
display("W2 naive cost: $μ ± $s")

W2_windowing_costs = train(ambiguity_radii, windowing_weights, windowing_parameters)
μ = mean(W2_windowing_costs)
s = sem(W2_windowing_costs)
display("W2 windowing cost: $μ ± $s")

W2_smoothing_costs = train(ambiguity_radii, smoothing_weights, smoothing_parameters)
μ = mean(W2_smoothing_costs)
s = sem(W2_smoothing_costs)
display("W2 smoothing cost: $μ ± $s")

W2_optimal_costs = train(ambiguity_radii, W2_optimal_weights, shift_bound_parameters)
μ = mean(W2_optimal_costs)
s = sem(W2_optimal_costs)
display("W2 optimal cost: $μ ± $s")
display(plot(1:history_length, reverse(W2_optimal_weights(history_length, 40, 1))))






function REMK_intersection_based_W2_newsvendor_order(ball_radii, ξ) 

    #return ξ[end]

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

    #termination_status(Problem)
    #primal_status(Problem)
    #primal_feasibility_report(Problem)
    #is_solved_and_feasible(Problem)

    #print(Problem)

    try
        return value(x)
    catch
        #display("catch")
        #return newsvendor_order(0, ξ, [1/K for k in 1:K])
        #return ξ[end]
        return REMK_intersection_based_W2_newsvendor_order(2*ball_radii,ξ)
    end
end

K = 30
function train(initial_ball_radii_parameters, shift_bound_parameters)

    costs = [zeros((length(initial_ball_radii_parameters),length(shift_bound_parameters))) for repetition in 1:repetitions]

    Threads.@threads for (initial_ball_radius_index, shift_bound_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(initial_ball_radii_parameters), eachindex(shift_bound_parameters))))

        ball_radii = reverse([initial_ball_radii_parameters[initial_ball_radius_index]+(k-1)*shift_bound_parameters[shift_bound_parameter_index] for k in 1:K])

        for repetition in 1:repetitions
            demand_samples = demand_sequences[repetition][1:history_length]
            order = REMK_intersection_based_W2_newsvendor_order(ball_radii, demand_samples)
            costs[repetition][initial_ball_radius_index, shift_bound_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])
        end
    end

    initial_ball_radius_index, shift_bound_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][initial_ball_radius_index, shift_bound_parameter_index] for repetition in 1:repetitions]
    display((initial_ball_radii_parameters[initial_ball_radius_index], shift_bound_parameters[shift_bound_parameter_index]))

    return minimal_costs
end


initial_ball_radii_parameters = LinRange(2000,6000,6)
shift_bound_parameters = LinRange(1000,2000,6)

intersection_based_costs = train(initial_ball_radii_parameters, shift_bound_parameters)
#intersection_based_costs = train([5200], [1200])
μ = mean(intersection_based_costs)
s = sem(intersection_based_costs)
display("intersection based cost: $μ ± $s")
