using  Random, Statistics, StatsBase, Distributions

number_of_consumers = 10000
D = number_of_consumers

Cu = 3 # Cost of underage.
Co = 1 # Cost of overage.

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
                                        # b(x)[i] + w[t,i]*ξ[t] + (1/4)*(1/λ)*w[t,i]^2 + z[t,i,:]'*d <= γ[t] <==>
                                        # w[t,i]^2 <= 2*(2*λ)*(γ[t] - b(x)[i] - w[t,i]*ξ[t] - z[t,i,:]'*d) <==>
                                        [2*λ; γ[t] - b(x)[i] - w[t,i]*ξ[t] - z[t,i,:]'*d; w[t,i]] in MathOptInterface.RotatedSecondOrderCone(3)
                                        a[i] - C'*z[t,i,:] == w[t,i]
                                  end)
        end
    end

    @objective(Problem, Min, ε*λ + weights'*γ)

    optimize!(Problem)

    try
        return value(x)
    catch
        #display("catch")
        return newsvendor_order(ε, ξ, weights)
    end
end

include("weights.jl")

Random.seed!(42)

shift_distribution = Uniform(-0.0005,0.0005)

initial_demand_probability = 0.1

repetitions = 1000
history_length = 100

demand_sequences = [zeros(history_length+1) for _ in 1:repetitions]
for repetition in 1:repetitions
    demand_probability = initial_demand_probability
    for t in 1:history_length+1
        demand_sequences[repetition][t] = rand(Binomial(number_of_consumers, demand_probability))
        demand_probability = min(max(demand_probability + rand(shift_distribution), 0), 1.0)
    end
end



using ProgressBars, IterTools
function parameter_fit(ambiguity_radii, compute_weights, weight_parameters)

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

    digits = 4

    return round(mean(minimal_costs), digits=digits), round(sem(minimal_costs), digits=digits), round(ambiguity_radii[ambiguity_radius_index], digits=digits), round(weight_parameters[weight_parameter_index], digits=digits)
end

ambiguity_radii = [LinRange(0.1,1,10); LinRange(2,10,9); LinRange(20,100,9); LinRange(200,1000,9)]
windowing_parameters = round.(Int, LinRange(1,history_length,40))
smoothing_parameters = [LinRange(0.001,0.01,10); LinRange(0.02,0.3,29)]
shift_bound_parameters = [LinRange(0.001,0.01,10); LinRange(0.02,0.1,9); LinRange(0.2,1,9); LinRange(2,10,9)]

#=
W₁_naive_cost, W₁_naive_sem, _, _ = parameter_fit([0], windowing_weights, [history_length])
display("W₁ naive: $W₁_naive_cost ± $W₁_naive_sem")


W₁_windowing_cost, W₁_windowing_sem, W₁_windowing_ε, W₁_windowing_t = parameter_fit([0], windowing_weights, windowing_parameters)
display("W₁ windowing: $W₁_windowing_t, $W₁_windowing_cost ± $W₁_windowing_sem")
W₁_windowing_t = round(Int, W₁_windowing_t)


W₁_smoothing_cost, W₁_smoothing_sem, W₁_smoothing_ε, W₁_smoothing_α = parameter_fit([0], smoothing_weights, smoothing_parameters)
display("W₁ smoothing: $W₁_smoothing_α, $W₁_smoothing_cost ± $W₁_smoothing_sem")


W₁_concentration_cost, W₁_concentration_sem, W₁_concentration_ε, W₁_concentration_ϱ = parameter_fit(ambiguity_radii, W₁_concentration_weights, shift_bound_parameters)
display("W₁ concentration: $W₁_concentration_ε, $W₁_concentration_ϱ, $W₁_concentration_cost ± $W₁_concentration_sem")
W₁_concentration_ε = round(Int, W₁_concentration_ε)


try
    println("Parameters &  & \$t=$W₁_windowing_t\$ & \$\\alpha=$W₁_smoothing_α\$ \\\\")
    println("Expected cost & \$$W₁_naive_cost \\pm $W₁_naive_sem\$ & \$$W₁_windowing_cost \\pm $W₁_windowing_sem\$ & \$$W₁_smoothing_cost \\pm $W₁_smoothing_sem\$\\\\")
catch
end
=#


function parameter_fit(ambiguity_radii, compute_weights, weight_parameters)

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

    digits = 4

    return round(mean(minimal_costs),digits=digits), round(sem(minimal_costs),digits=digits), round(ambiguity_radii[ambiguity_radius_index],digits=digits), round(weight_parameters[weight_parameter_index],digits=digits)
end

ambiguity_radii = LinRange(10,100,10)
windowing_parameters = round.(Int, LinRange(10,history_length,31))
smoothing_parameters = LinRange(0.01,0.3,30)
shift_bound_parameters = LinRange(0.1,1,10)


#=
W₂_naive_cost, W₂_naive_sem, W₂_naive_ε, _ = parameter_fit(ambiguity_radii, windowing_weights, history_length)
display("W₂ naive: $W₂_naive_ε, $W₂_naive_cost ± $W₂_naive_sem")
W₂_naive_ε = round(Int, W₂_naive_ε)

W₂_windowing_cost, W₂_windowing_sem, W₂_windowing_ε, W₂_windowing_t = parameter_fit(ambiguity_radii, windowing_weights, windowing_parameters)
display("W₂ windowing: $W₂_windowing_ε, $W₂_windowing_t, $W₂_windowing_cost ± $W₂_windowing_sem")
W₂_windowing_ε = round(Int, W₂_windowing_ε)
W₂_windowing_t = round(Int, W₂_windowing_t)

W₂_smoothing_cost, W₂_smoothing_sem, W₂_smoothing_ε, W₂_smoothing_α = parameter_fit(ambiguity_radii, smoothing_weights, smoothing_parameters)
display("W₂ smoothing: $W₂_smoothing_ε, $W₂_smoothing_α, $W₂_smoothing_cost ± $W₂_smoothing_sem")
W₂_smoothing_ε = round(Int, W₂_smoothing_ε)
=#

W₂_concentration_cost, W₂_concentration_sem, W₂_concentration_ε, W₂_concentration_ϱ = parameter_fit(ambiguity_radii, W₂_concentration_weights, shift_bound_parameters)
display("W₂ concentration: $W₂_concentration_ε, $W₂_concentration_ϱ, $W₂_concentration_cost ± $W₂_concentration_sem")
W₂_concentration_ε = round(Int, W₂_concentration_ε)


try
    println("Parameters & \$\\varepsilon=$W₂_naive_ε\$ & \$\\varepsilon=$W₂_windowing_ε\$, \$t=$W₂_windowing_t\$ & \$\\varepsilon=$W₂_smoothing_ε\$, \$\\alpha=$W₂_smoothing_α\$ & \$\\varepsilon=$W₂_concentration_ε\$, \$\\varrho=$W₂_concentration_ϱ\$ & \$ \$ \\\\")
    println("Expected cost & \$$W₂_naive_cost \\pm $W₂_naive_sem\$ & \$$W₂_windowing_cost \\pm $W₂_windowing_sem\$ & \$$W₂_smoothing_cost \\pm $W₂_smoothing_sem\$ & \$$W₂_concentration_cost \\pm $W₂_concentration_sem\$ & \$ \$ \\\\")
catch
end



function REMK_intersection_based_W₂_newsvendor_order(ball_radii, ξ, empty_counter) 

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
                                    # <==> b(x)[i] + sum(w[i,k]*ξ[k] + s[i,k] for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K),      (1/4)*(1/λ[K])*w[i,k]^2 <= s[i,k] for all i,k
                                    # <==> b(x)[i] + sum(w[i,k]*ξ[k] + s[i,k] for k in 1:K) + z[i,:]'*d <= sum(γ[k] for k in 1:K),      [2*λ[k]; s[i,k]; w[i,k]] in MathOptInterface.RotatedSecondOrderCone(3) for all i,k
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

    try
        return value(x), empty_counter
    catch
        return REMK_intersection_based_W₂_newsvendor_order(2*ball_radii,ξ,1)
    end
end

K = history_length
function parameter_fit(initial_ball_radii_parameters, shift_bound_parameters)

    costs = [zeros((length(initial_ball_radii_parameters),length(shift_bound_parameters))) for repetition in 1:repetitions]
    empty_counters = [zeros((length(initial_ball_radii_parameters),length(shift_bound_parameters))) for repetition in 1:repetitions]

    for (initial_ball_radius_index, shift_bound_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(initial_ball_radii_parameters), eachindex(shift_bound_parameters))))

        ball_radii = reverse([initial_ball_radii_parameters[initial_ball_radius_index]+(k-1)*shift_bound_parameters[shift_bound_parameter_index] for k in 1:K])

        Threads.@threads for repetition in 1:repetitions
            demand_samples = demand_sequences[repetition][1:history_length]
            order, empty_counter = REMK_intersection_based_W₂_newsvendor_order(ball_radii, demand_samples, 0)
            costs[repetition][initial_ball_radius_index, shift_bound_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])
            empty_counters[repetition][initial_ball_radius_index, shift_bound_parameter_index] = empty_counter
        
        end
    end

    initial_ball_radius_index, shift_bound_parameter_index = Tuple(argmin(mean(costs)))
    minimal_costs = [costs[repetition][initial_ball_radius_index, shift_bound_parameter_index] for repetition in 1:repetitions]
    empty_frequency = mean([empty_counters[repetition][initial_ball_radius_index, shift_bound_parameter_index] for repetition in 1:repetitions])

    digits = 4

    return round(mean(minimal_costs), digits=digits), round(sem(minimal_costs), digits=digits), round(initial_ball_radii_parameters[initial_ball_radius_index], digits=digits), round(shift_bound_parameters[shift_bound_parameter_index], digits=digits), empty_frequency
end

initial_ball_radii_parameters = [LinRange(100,1000,4); LinRange(4000,10000,3); LinRange(40000,100000,3)]
shift_bound_parameters = [LinRange(1,10,4); LinRange(40,100,3); LinRange(400,1000,3)]

#intersection_based_cost, intersection_based_sem, intersection_based_ε, intersection_based_ϱ, empty_frequency = parameter_fit(initial_ball_radii_parameters, shift_bound_parameters)
#display("W₂ intersection: $intersection_based_ε, $intersection_based_ϱ, $intersection_based_cost ± $intersection_based_sem, $empty_frequency")






























































if false

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
            stack(demand_sequences[2:100])[1:end-1,:], 
            xlims = (0,history_length+1),
            xlabel = "Time", 
            ylabel = "Demand",
            labels = nothing, 
            #linecolor = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5]],
            #markercolor = palette(:tab10)[1],
            #markershape = :circle,
            color = palette(:tab10)[1],
            alpha = 0.03,
            #linestyle = :auto,
            #markersize = 4, 
            #markerstrokewidth = 1,
            #markerstrokecolor = :black,
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 3pt, 
            leftmargin = 3pt,
            )

    plot!(1:history_length, 
            stack(demand_sequences[1])[1:end-1,:], 
            labels = nothing, 
            #linecolor = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5]],
            markercolor = palette(:tab10)[1],
            markershape = :circle,
            color = palette(:tab10)[1],
            alpha = 1.0,
            #linestyle = :auto,
            markersize = 4, 
            markerstrokewidth = 1,
            markerstrokecolor = :black,
            )

    display(plt)

    #savefig(plt, "figures/demand_sequences.pdf")



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
    W₁_weights = reverse(W₁_concentration_weights(history_length, W₁_concentration_ε, W₁_concentration_ϱ))

    W₁_windowing_t = round(Int, W₁_windowing_t)

    plt = plot(1:history_length, stack([W₁_windowing_weights, W₁_smoothing_weights, W₁_weights]), 
            xlabel = "Time", 
            ylabel = "Probability",
            xlims = (0,history_length+1),
            #legend = nothing,
            labels = ["\$t=$W₁_windowing_t\$" "\$α=$W₁_smoothing_α\$" "\$ε=$W₁_concentration_ε\$, \$ϱ=$W₁_concentration_ϱ\$"],
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

    #savefig(plt, "figures/W1-weights.pdf")




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
    W₂_weights = reverse(W₂_concentration_weights(history_length, W₂_concentration_ε, W₂_concentration_ϱ))

    W₂_windowing_t = round(Int, W₂_windowing_t)

    plt = plot(1:history_length, stack([W₂_windowing_weights, W₂_smoothing_weights, W₂_weights]), 
            xlabel = "Time", 
            ylabel = "Probability",
            xlims = (0,history_length+1),
            #legend = nothing,
            labels = ["\$ε=$W₂_windowing_ε\$, \$t=$W₂_windowing_t\$" "\$ε=$W₂_smoothing_ε\$, \$α=$W₂_smoothing_α\$" "\$ε=$W₂_concentration_ε\$, \$ϱ=$W₂_concentration_ϱ\$"],
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

    #savefig(plt, "figures/W2-weights.pdf")

end