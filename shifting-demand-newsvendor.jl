using  Random, Statistics, StatsBase, Distributions

number_of_consumers = 10000
D = number_of_consumers

Cu = 1 # Cost of underage.
Co = 2/3 # Cost of overage.

newsvendor_order(ε, ξ, weights) = quantile(ξ, Weights(weights), Cu/(Co+Cu))

using JuMP, MathOptInterface
using Gurobi
env = Gurobi.Env() 
GRBsetintparam(env, "OutputFlag", 0)
GRBsetintparam(env, "BarHomogeneous", 1)
optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

function W1_newsvendor_order(ε, ξ, weights) 

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
                                 λ
                                 s[t=1:T]
                                 γ[t=1:T,i=1:2,j=1:2] >= 0
                                 z[t=1:T,i=1:2] 
                        end)

    for t in 1:T
        for i in 1:2
            @constraints(Problem, begin
                                        b(x)[i] + a[i]*ξ[t] + γ[t,i,:]'*(d-C*ξ[t]) <= s[t]
                                        z[t,i] <= λ
                                        C'*γ[t,i,:] - a[i] <= z[t,i]
                                       -C'*γ[t,i,:] + a[i] <= z[t,i]
                                  end)
        end
    end

    @objective(Problem, Min, ε*λ + weights'*s)

    optimize!(Problem)

    try
        return value(x)
    catch
        display("catch")
        return newsvendor_order(ε, ξ, weights)

    end

end

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

newsvendor_loss(x,ξ) = Cu*max(ξ-x,0) + Co*max(x-ξ,0)



include("weights.jl")

Random.seed!(42)

shift_distribution = Uniform(-0.002,0.002)

initial_demand_probability = 0.1

repetitions = 1000
history_length = 30

windowing_parameters = round.(Int, LinRange(3,history_length,6))
smoothing_parameters = LinRange(0.0001,0.4,6)
ambiguity_radii = LinRange(1e1,1e3,6) #LinRange(1e1,1e2,21) #LinRange(1e1,1.5e2,21)
shift_bound_parameters = LinRange(1.0,20.0,6) #[0.05,0.1,0.15,0.2] #LinRange(1e-1,1e-0,11) #[0.1] #[0,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3]


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
            order = W2_newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
            costs[repetition][ambiguity_radius_index, weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])

        end
    end

    ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][ambiguity_radius_index, weight_parameter_index] for repetition in 1:repetitions]
    display((ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index]))

    return minimal_costs
end

#=
naive_costs = train(ambiguity_radii, windowing_weights, [history_length])
μ = mean(naive_costs)
s = sem(naive_costs)
display("SAA cost: $μ ± $s")

windowing_costs = train(ambiguity_radii, windowing_weights, windowing_parameters)
μ = mean(windowing_costs)
s = sem(windowing_costs)
display("Windowing cost: $μ ± $s")
=#

#=
smoothing_costs = train(ambiguity_radii, smoothing_weights, smoothing_parameters)
μ = mean(smoothing_costs)
s = sem(smoothing_costs)
display("smoothing cost: $μ ± $s")
=#

#=
optimal_costs = train(ambiguity_radii, optimal_weights, shift_bound_parameters)
μ = mean(optimal_costs)
s = sem(optimal_costs)
display("optimal cost: $μ ± $s")
=#

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


function intersection_based_W2_newsvendor_order(ball_radii, ξ, ball_weights) 

    indices_to_keep = [maximum(stack(ball_weights)'[:,i]) for i in 1:length(ξ)] .>= 1e-3
    ξ = ξ[indices_to_keep]
    for ball_weight_index in eachindex(ball_weights)
        ball_weights[ball_weight_index] = ball_weights[ball_weight_index][indices_to_keep]
        ball_weights[ball_weight_index] .= ball_weights[ball_weight_index]/sum(ball_weights[ball_weight_index])
    end

    K = length(ball_radii)
    T = length(ξ)

    combinations = collect(IterTools.product(Tuple(1:length(ξ) for k in 1:K)...))
    A = length(combinations)

    Problem = Model(optimizer)

    C = [-1; 1]
    d = [0, D]
    g = [Cu,-Co]
    b(x) = [-Cu*x, Co*x]

    @variables(Problem, begin
                            D >= x >= 0
                                 λ[k=1:K] >= 0
                                 γ[k=1:K,t=1:T]
                                 z[a=1:A,i=1:2,j=1:2] >= 0
                                 w[a=1:A,i=1:2,k=1:K]
                                 s[a=1:A,i=1:2,k=1:K]
                        end)

    for a in 1:A
        for i in 1:2
            @constraints(Problem, begin
                                        # b(x)[i] + sum(w[a,i,k]*ξ[combinations[a][k]] + (1/4)*(1/λ[k])*w[a,i,k]^2 for k in 1:K) + z[a,i,:]'*d <= sum(γ[k,combinations[a][k]] for k in 1:K)
                                        # <==> b(x)[i] + sum(w[a,i,k]*ξ[combinations[a][k]] + s[a,i,k] for k in 1:K) + z[a,i,:]'*d <= sum(γ[k,combinations[a][k]] for k in 1:K),      (1/4)*(1/λ[K])*w[a,i,k]^2 <= s[a,i,k]
                                        # <==> b(x)[i] + sum(w[a,i,k]*ξ[combinations[a][k]] + s[a,i,k] for k in 1:K) + z[a,i,:]'*d <= sum(γ[k,combinations[a][k]] for k in 1:K),      [2*λ[k]; s[a,i,k]; w[a,i,k]] in MathOptInterface.RotatedSecondOrderCone(3) for t in 1:T
                                        b(x)[i] + sum(w[a,i,k]*ξ[combinations[a][k]] + s[a,i,k] for k in 1:K) + z[a,i,:]'*d <= sum(γ[k,combinations[a][k]] for k in 1:K)
                                        g[i] - C'*z[a,i,:] == sum(w[a,i,k] for k in 1:K)
                                  end)
        end
    end

    for a in 1:A
        for i in 1:2
            for k in 1:K
                @constraints(Problem, begin
                                            [2*λ[k]; s[a,i,k]; w[a,i,k]] in MathOptInterface.RotatedSecondOrderCone(3) 
                                      end)
            end
        end
    end

    @objective(Problem, Min, sum(ball_radii[k]*λ[k] for k in 1:K) + sum(sum(ball_weights[k][t]*γ[k,t] for t in 1:T) for k in 1:K))

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
        return newsvendor_order(0, ξ, [1/T for t in 1:T])
    end
end


ball_radii = [LinRange(1e3,2e3,2).+1e1, 2e2] #[LinRange(1e2,1e4,2), LinRange(1e1,1e4,2)]
shift_bound_parameters = [1e1] #LinRange(1.0,20.0,2)

function train(ball_radii, compute_weights, weight_parameters)

    precomputed_weights = [[[zeros(history_length) for weight_parameter_index in eachindex(weight_parameters)] for ball_radius_index in eachindex(ball_radii[ball_radii_index])] for ball_radii_index in eachindex(ball_radii)]

    #display(precomputed_weights)

    for ball_radii_index in eachindex(ball_radii)
        for ball_radius_index in eachindex(ball_radii[ball_radii_index])
            for weight_parameter_index in eachindex(weight_parameters)
                precomputed_weights[ball_radii_index][ball_radius_index][weight_parameter_index] = 
                    [compute_weights(history_length-(length(ball_radii)-ball_radii_index), ball_radii[ball_radii_index][ball_radius_index], weight_parameters[weight_parameter_index]); zeros(length(ball_radii)-ball_radii_index)]
            end
        end
    end

    ball_radii_indices = vec(collect(IterTools.product([eachindex(ball_radii[ball_radii_index]) for ball_radii_index in eachindex(ball_radii)]...)))
    
    #display(ball_radii_indices)

    costs = [zeros((length(ball_radii_indices),length(weight_parameters))) for repetition in 1:repetitions]

    Threads.@threads for (ball_radii_indices_index, weight_parameter_index, repetition) in ProgressBar(collect(IterTools.product(eachindex(ball_radii_indices), eachindex(weight_parameters), 1:repetitions)))
        #display(ball_radii_index)
        weights = [precomputed_weights[k][ball_radii_indices[ball_radii_indices_index][k]][weight_parameter_index] for k in eachindex(ball_radii)]

        #for repetition in 1:repetitions
        demand_samples = demand_sequences[repetition][1:history_length]
        order = intersection_based_W2_newsvendor_order([ball_radii[k][ball_radii_indices[ball_radii_indices_index][k]] for k in eachindex(ball_radii)], demand_samples, weights)
        costs[repetition][ball_radii_indices_index, weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])

        #end
    end

    ball_radii_index, weight_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][ball_radii_index, weight_parameter_index] for repetition in 1:repetitions]
    display((ball_radii_indices[ball_radii_index], weight_parameters[weight_parameter_index]))

    return minimal_costs
end

#=
intersection_based_costs = train(ball_radii, optimal_weights, shift_bound_parameters)
μ = mean(intersection_based_costs)
s = sem(intersection_based_costs)
display("intersection based cost: $μ ± $s")
=#

#=
intersection_based_costs = train(ball_radii, optimal_weights, shift_bound_parameters)
#intersection_based_costs = train(ball_radii, windowing_weights, [1])
μ = mean(intersection_based_costs)
s = sem(intersection_based_costs)
display("intersection based cost: $μ ± $s")
=#










function windowing_intersection_based_W2_newsvendor_order(ball_radii, ξ) 

    K = length(ball_radii)
    ξ = ξ[end-K+1:end]
    #T = length(ξ)

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
                                    # b(x)[i] + sum(w[a,i,k]*ξ[combinations[a][k]] + (1/4)*(1/λ[k])*w[a,i,k]^2 for k in 1:K) + z[a,i,:]'*d <= sum(γ[k,combinations[a][k]] for k in 1:K)
                                    # <==> b(x)[i] + sum(w[a,i,k]*ξ[combinations[a][k]] + s[a,i,k] for k in 1:K) + z[a,i,:]'*d <= sum(γ[k,combinations[a][k]] for k in 1:K),      (1/4)*(1/λ[K])*w[a,i,k]^2 <= s[a,i,k]
                                    # <==> b(x)[i] + sum(w[a,i,k]*ξ[combinations[a][k]] + s[a,i,k] for k in 1:K) + z[a,i,:]'*d <= sum(γ[k,combinations[a][k]] for k in 1:K),      [2*λ[k]; s[a,i,k]; w[a,i,k]] in MathOptInterface.RotatedSecondOrderCone(3) for t in 1:T
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
        display("catch")
        #return newsvendor_order(0, ξ, [1/K for k in 1:K])
        return windowing_intersection_based_W2_newsvendor_order(2*ball_radii,ξ)
    end
end


initial_ball_radii_parameters = 1000*[1e2, 9e1, 8e1, 7e1, 6e1, 5e1, 4e1, 3e1, 2e1, 1e1] #LinRange(1e5,3e5,3), LinRange(1e4,3e4,3), LinRange(1e3,3e3,3), LinRange(5e1,2e2,3), LinRange(1e1,3e1,3)]
shift_bound_parameters = 

function train(initial_ball_radii_parameters, shift_bound_parameters)

    ball_radii_indices = vec(collect(IterTools.product([eachindex(ball_radii[ball_radii_index]) for ball_radii_index in eachindex(ball_radii)]...)))
    costs = [zeros((length(ball_radii_indices))) for repetition in 1:repetitions]

    Threads.@threads for (ball_radii_indices_index, repetition) in ProgressBar(collect(IterTools.product(eachindex(ball_radii_indices), 1:repetitions)))
        demand_samples = demand_sequences[repetition][1:history_length]
        order = windowing_intersection_based_W2_newsvendor_order([ball_radii[k][ball_radii_indices[ball_radii_indices_index][k]] for k in eachindex(ball_radii)], demand_samples)
        costs[repetition][ball_radii_indices_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])

    end

    ball_radii_index = argmin(mean(costs))
    minimal_costs = [costs[repetition][ball_radii_index] for repetition in 1:repetitions]
    display(ball_radii_indices[ball_radii_index])

    return minimal_costs
end

intersection_based_costs = train(ball_radii)
#intersection_based_costs = train(ball_radii, windowing_weights, [1])
μ = mean(intersection_based_costs)
s = sem(intersection_based_costs)
display("intersection based cost: $μ ± $s")