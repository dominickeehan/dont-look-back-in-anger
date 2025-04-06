using  Random, Statistics, StatsBase, Distributions

number_of_consumers = 10000
D = number_of_consumers

Cu = 1 # Cost of underage.
Co = 2/3 # Cost of overage.

newsvendor_loss(x,ξ) = Cu*max(ξ-x,0) + Co*max(x-ξ,0)

newsvendor_order(ε, ξ, weights) = quantile(ξ, Weights(weights), Cu/(Co+Cu))
W₁_newsvendor_order(ε, ξ, weights) = newsvendor_order(ε, ξ, weights)

using JuMP, MathOptInterface
using Gurobi
env = Gurobi.Env() 
GRBsetintparam(env, "OutputFlag", 0)
GRBsetintparam(env, "BarHomogeneous", 1)
optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

function W₂_newsvendor_order(ε, ξ, weights) 

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
        stack(demand_sequences[1])[1:end-1,:], 
        xlims = (0,history_length+1),
        xlabel = "Time", 
        ylabel = "Demand",
        labels = nothing, 
        #linecolor = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5]],
        markercolors = [palette(:tab10)[1]],
        markershapes = [:circle],
        colors = [palette(:tab10)[1]],
        linewidth = 1,
        #alpha = 1,
        #linestyle = :auto,
        markersize = 4, 
        markerstrokewidth = 1,
        markerstrokecolor = :black,
        topmargin = 0pt, 
        rightmargin = 0pt,
        bottommargin = 3pt, 
        leftmargin = 3pt,
        )

display(plt)

savefig(plt, "figures/demand_sequence.pdf")


using ProgressBars, IterTools
function train(ambiguity_radii, compute_weights, weight_parameters)

    costs = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in 1:repetitions]

    Threads.@threads for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
        weights = compute_weights(history_length, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index])

        for repetition in 1:repetitions
            demand_samples = demand_sequences[repetition][1:history_length]
            order = W₁_newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
            costs[repetition][ambiguity_radius_index, weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])

        end
    end

    ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][ambiguity_radius_index, weight_parameter_index] for repetition in 1:repetitions]

    return round(mean(minimal_costs), digits=2), round(sem(minimal_costs), digits=2), round(ambiguity_radii[ambiguity_radius_index], digits=2), round(weight_parameters[weight_parameter_index], digits=2)
end

windowing_parameters = round.(Int, LinRange(3,history_length,11))
smoothing_parameters = LinRange(0.0001,0.4,11)
ambiguity_radii = LinRange(60,200,11) # Only matters for optimal costs.
shift_bound_parameters = LinRange(2,20,11)

W₁_naive_cost, W₁_naive_sem, _, _ = train([0], windowing_weights, [history_length])
display("W₁ naive: $W₁_naive_cost ± $W₁_naive_sem")

W₁_windowing_cost, W₁_windowing_sem, W₁_windowing_ε, W₁_windowing_t = train([0], windowing_weights, windowing_parameters)
display("W₁ windowing: $W₁_windowing_t, $W₁_windowing_cost ± $W₁_windowing_sem")

W₁_smoothing_cost, W₁_smoothing_sem, W₁_smoothing_ε, W₁_smoothing_α = train([0], smoothing_weights, smoothing_parameters)
display("W₁ smoothing: $W₁_smoothing_α, $W₁_smoothing_cost ± $W₁_smoothing_sem")

W₁_optimal_cost, W₁_optimal_sem, W₁_optimal_ε, W₁_optimal_ϱ = train(ambiguity_radii, W₁_optimal_weights, shift_bound_parameters)
display("W₁ optimal: $W₁_optimal_ε, $W₁_optimal_ϱ, $W₁_optimal_cost ± $W₁_optimal_sem")

println("Parameters &  & \$t=$W₁_windowing_t\$ & \$\\alpha=$W₁_smoothing_α\$ & \$\\varepsilon=$W₁_optimal_ε\$, \$\\varrho=$W₁_optimal_ϱ\$ \\\\")
println("Expected cost & \$$W₁_naive_cost \\pm $W₁_naive_sem\$ & \$$W₁_windowing_cost \\pm $W₁_windowing_sem\$ & \$$W₁_smoothing_cost \\pm $W₁_smoothing_sem\$ & \$$W₁_optimal_cost \\pm $W₁_optimal_sem\$\\\\")


default() # Reset plot defaults.

gr(size = (600,400))

font_family = "Computer Modern"
primary_font = Plots.font(font_family, pointsize = 17)
secondary_font = Plots.font(font_family, pointsize = 11)
legend_font = Plots.font(font_family, pointsize = 15)

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

W₁_windowing_weights = reverse(windowing_weights(history_length, [0], round(Int, W₁_windowing_t)))
W₁_smoothing_weights = reverse(smoothing_weights(history_length, [0], W₁_smoothing_α))
W₁_weights = reverse(W₁_optimal_weights(history_length, W₁_optimal_ε, W₁_optimal_ϱ))

W₁_windowing_t = round(Int, W₁_windowing_t)

plt = plot(1:history_length, stack([W₁_windowing_weights, W₁_smoothing_weights, W₁_weights]), 
        xlabel = "Time", 
        ylabel = "Probability",
        xlims = (0,history_length+1),
        #legend = nothing,
        labels = ["\$t=$W₁_windowing_t\$" "\$α=$W₁_smoothing_α\$" "\$ε=$W₁_optimal_ε\$, \$ϱ=$W₁_optimal_ϱ\$"],
        colors = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3]],
        #markershapes = [:circle :diamond :hexagon],
        seriestypes = [:steppre :line :line],
        alpha = 1,
        #markersize = 2,
        #linestyles = :auto,
        linewidth = 1,
        topmargin = 0pt, 
        rightmargin = 0pt,
        bottommargin = 3pt, 
        leftmargin = 3pt)

display(plt);

savefig(plt, "figures/W1-weights.pdf")






function train(ambiguity_radii, compute_weights, weight_parameters)

    costs = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in 1:repetitions]

    Threads.@threads for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
        weights = compute_weights(history_length, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index])

        for repetition in 1:repetitions
            demand_samples = demand_sequences[repetition][1:history_length]
            order = W₂_newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
            costs[repetition][ambiguity_radius_index, weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])

        end
    end

    ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][ambiguity_radius_index, weight_parameter_index] for repetition in 1:repetitions]

    return round(mean(minimal_costs), digits=2), round(sem(minimal_costs), digits=2), round(ambiguity_radii[ambiguity_radius_index], digits=2), round(weight_parameters[weight_parameter_index], digits=2)
end

ambiguity_radii = LinRange(60,600,11)
windowing_parameters = round.(Int, LinRange(3,history_length,11))
smoothing_parameters = LinRange(0.0001,0.4,11)
shift_bound_parameters = LinRange(1,20,11)


W₂_naive_cost, W₂_naive_sem, W₂_naive_ε, _ = train(ambiguity_radii, windowing_weights, history_length)
display("W₂ naive: $W₂_naive_ε, $W₂_naive_cost ± $W₂_naive_sem")
W₂_naive_ε = round(Int, W₂_naive_ε)

W₂_windowing_cost, W₂_windowing_sem, W₂_windowing_ε, W₂_windowing_t = train(ambiguity_radii, windowing_weights, windowing_parameters)
display("W₂ windowing: $W₂_windowing_ε, $W₂_windowing_t, $W₂_windowing_cost ± $W₂_windowing_sem")
W₂_windowing_ε = round(Int, W₂_windowing_ε)

W₂_smoothing_cost, W₂_smoothing_sem, W₂_smoothing_ε, W₂_smoothing_α = train(ambiguity_radii, smoothing_weights, smoothing_parameters)
display("W₂ smoothing: $W₂_smoothing_ε, $W₂_smoothing_α, $W₂_smoothing_cost ± $W₂_smoothing_sem")
W₂_smoothing_ε = round(Int, W₂_smoothing_ε)

W₂_optimal_cost, W₂_optimal_sem, W₂_optimal_ε, W₂_optimal_ϱ = train(ambiguity_radii, W₂_optimal_weights, shift_bound_parameters)
display("W₂ optimal: $W₂_optimal_ε, $W₂_optimal_ϱ, $W₂_optimal_cost ± $W₂_optimal_sem")
W₂_optimal_ε = round(Int, W₂_optimal_ε)

println("Parameters & \$\\varepsilon=$W₂_naive_ε\$ & \$\\varepsilon=$W₂_windowing_ε\$, \$t=$W₂_windowing_t\$ & \$\\varepsilon=$W₂_smoothing_ε\$, \$\\alpha=$W₂_smoothing_α\$ & \$\\varepsilon=$W₂_optimal_ε\$, \$\\varrho=$W₂_optimal_ϱ\$ \\\\")
println("Expected cost & \$$W₂_naive_cost \\pm $W₂_naive_sem\$ & \$$W₂_windowing_cost \\pm $W₂_windowing_sem\$ & \$$W₂_smoothing_cost \\pm $W₂_smoothing_sem\$ & \$$W₂_optimal_cost \\pm $W₂_optimal_sem\$\\\\")



default() # Reset plot defaults.

gr(size = (600,400))

font_family = "Computer Modern"
primary_font = Plots.font(font_family, pointsize = 17)
secondary_font = Plots.font(font_family, pointsize = 11)
legend_font = Plots.font(font_family, pointsize = 15)

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

W₂_windowing_weights = reverse(windowing_weights(history_length, [0], round(Int, W₂_windowing_t)))
W₂_smoothing_weights = reverse(smoothing_weights(history_length, [0], W₂_smoothing_α))
W₂_weights = reverse(W₂_optimal_weights(history_length, W₂_optimal_ε, W₂_optimal_ϱ))

W₂_windowing_t = round(Int, W₂_windowing_t)

plt = plot(1:history_length, stack([W₂_windowing_weights, W₂_smoothing_weights, W₂_weights]), 
        xlabel = "Time", 
        ylabel = "Probability",
        xlims = (0,history_length+1),
        #legend = nothing,
        labels = ["\$ε=$W₂_windowing_ε\$, \$t=$W₂_windowing_t\$" "\$ε=$W₂_smoothing_ε\$, \$α=$W₂_smoothing_α\$" "\$ε=$W₂_optimal_ε\$, \$ϱ=$W₂_optimal_ϱ\$"],
        colors = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3]],
        #markershapes = [:circle :diamond :hexagon],
        seriestypes = [:steppre :line :line],
        alpha = 1,
        #markersize = 2,
        #linestyles = :auto,
        linewidth = 1,
        topmargin = 0pt, 
        rightmargin = 0pt,
        bottommargin = 3pt, 
        leftmargin = 3pt)

display(plt);

savefig(plt, "figures/W2-weights.pdf")







function REMK_intersection_based_W₂_newsvendor_order(ball_radii, ξ) 

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
        return REMK_intersection_based_W₂_newsvendor_order(2*ball_radii,ξ)
    end
end

K = history_length
function train(initial_ball_radii_parameters, shift_bound_parameters)

    costs = [zeros((length(initial_ball_radii_parameters),length(shift_bound_parameters))) for repetition in 1:repetitions]

    Threads.@threads for (initial_ball_radius_index, shift_bound_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(initial_ball_radii_parameters), eachindex(shift_bound_parameters))))

        ball_radii = reverse([initial_ball_radii_parameters[initial_ball_radius_index]+(k-1)*shift_bound_parameters[shift_bound_parameter_index] for k in 1:K])

        for repetition in 1:repetitions
            demand_samples = demand_sequences[repetition][1:history_length]
            order = REMK_intersection_based_W₂_newsvendor_order(ball_radii, demand_samples)
            costs[repetition][initial_ball_radius_index, shift_bound_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])
        end
    end

    initial_ball_radius_index, shift_bound_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][initial_ball_radius_index, shift_bound_parameter_index] for repetition in 1:repetitions]

    return round(mean(minimal_costs), digits=2), round(sem(minimal_costs), digits=2), round(initial_ball_radii_parameters[initial_ball_radius_index], digits=2), round(shift_bound_parameters[shift_bound_parameter_index], digits=2)
end

initial_ball_radii_parameters = LinRange(20,200,11)
shift_bound_parameters = LinRange(1,40,11)

intersection_based_cost, intersection_based_sem, intersection_based_ε, intersection_based_ϱ  = train(initial_ball_radii_parameters, shift_bound_parameters)
#intersection_based_cost, intersection_based_sem, intersection_based_ε, intersection_based_ϱ = train([2600], [400])
display("W₂ intersection: $intersection_based_ε, $intersection_based_ϱ, $intersection_based_cost ± $intersection_based_sem")




































































#=










Random.seed!(42)

shift_distribution = Dirac(0)#Uniform(-0.001,0.001)

initial_demand_probability = 0.1

repetitions = 10000
history_length = 100

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
            order = W₁_newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
            costs[repetition][ambiguity_radius_index, weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])

        end
    end

    ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][ambiguity_radius_index, weight_parameter_index] for repetition in 1:repetitions]
    display((ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index]))

    return minimal_costs
end

ambiguity_radii = LinRange(10,100,11) # Only matters for optimal costs.
shift_bound_parameters = LinRange(1,10,11)


W₁_naive_costs = train([0], windowing_weights, [history_length])
μ = mean(W₁_naive_costs)
s = sem(W₁_naive_costs)
display("W₁ naive cost: $μ ± $s")

W₁_optimal_costs = train(ambiguity_radii, W₁_optimal_weights, shift_bound_parameters)
μ = mean(W₁_optimal_costs)
s = sem(W₁_optimal_costs)
display("W₁ optimal cost: $μ ± $s")
display(plot(1:history_length, reverse(W₁_optimal_weights(history_length, 21.1, 2.6))))




function train(ambiguity_radii, compute_weights, weight_parameters)

    costs = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in 1:repetitions]

    Threads.@threads for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
        weights = compute_weights(history_length, ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index])

        for repetition in 1:repetitions
            demand_samples = demand_sequences[repetition][1:history_length]
            order = W₂_newsvendor_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights)
            costs[repetition][ambiguity_radius_index, weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])

        end
    end

    ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][ambiguity_radius_index, weight_parameter_index] for repetition in 1:repetitions]
    display((ambiguity_radii[ambiguity_radius_index], weight_parameters[weight_parameter_index]))

    return minimal_costs
end

ambiguity_radii = LinRange(100,500,11)
shift_bound_parameters = LinRange(1,10,11)

#=
W₂_optimal_costs = train(ambiguity_radii, W₂_optimal_weights, shift_bound_parameters)
μ = mean(W₂_optimal_costs)
s = sem(W₂_optimal_costs)
display("W₂ optimal cost: $μ ± $s")
display(plot(1:history_length, reverse(W₂_optimal_weights(history_length, 40, 1))))
=#


function REMK_intersection_based_W₂_newsvendor_order(ball_radii, ξ) 

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
        return REMK_intersection_based_W₂_newsvendor_order(2*ball_radii,ξ)
    end
end

K = history_length
function train(initial_ball_radii_parameters, shift_bound_parameters)

    costs = [zeros((length(initial_ball_radii_parameters),length(shift_bound_parameters))) for repetition in 1:repetitions]

    Threads.@threads for (initial_ball_radius_index, shift_bound_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(initial_ball_radii_parameters), eachindex(shift_bound_parameters))))

        ball_radii = reverse([initial_ball_radii_parameters[initial_ball_radius_index]+(k-1)*shift_bound_parameters[shift_bound_parameter_index] for k in 1:K])

        for repetition in 1:repetitions
            demand_samples = demand_sequences[repetition][1:history_length]
            order = REMK_intersection_based_W₂_newsvendor_order(ball_radii, demand_samples)
            costs[repetition][initial_ball_radius_index, shift_bound_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])
        end
    end

    initial_ball_radius_index, shift_bound_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][initial_ball_radius_index, shift_bound_parameter_index] for repetition in 1:repetitions]
    display((initial_ball_radii_parameters[initial_ball_radius_index], shift_bound_parameters[shift_bound_parameter_index]))

    return minimal_costs
end


initial_ball_radii_parameters = LinRange(1000,2000,11)
shift_bound_parameters = LinRange(0,100,11)

#intersection_based_costs = train(initial_ball_radii_parameters, shift_bound_parameters)
intersection_based_costs = train([1900], [0.0])
μ = mean(intersection_based_costs)
s = sem(intersection_based_costs)
display("intersection based cost: $μ ± $s")


=#


