using Random, Statistics, StatsBase, Distributions
using ProgressBars, IterTools

include("weights.jl")
include("newsvendor-optimizations.jl")

repetitions = 100 #200 #200 # 100, 200
history_length = 70 # 10, 70, 100

function expected_newsvendor_cost(order, demand_probability)

    a = cdf(Binomial(D-1,demand_probability), order-1)
    b = cdf(Binomial(D,demand_probability), order)

    expected_underage_cost = Cu * (D*demand_probability*(1-a) - order*(1-b))
    expected_overage_cost = Co * (order*b - D*demand_probability*a)

    return expected_underage_cost + expected_overage_cost

end

#exp10.(LinRange(log10(1),log10(10),7))


#drifts = [1e-4, 2.1544e-4, 4.6416e-4, 1e-3, 2.1544e-3, 4.6416e-3, 1e-2, 2.1544e-2, 4.6416e-2, 6.8129e-2, 1e-1]
#drifts = [1e-4, 2.1544e-4, 4.6416e-4, 1e-3, 2.1544e-3, 4.6416e-3, 1e-2, 2.1544e-2, 4.6416e-2, 6.8129e-2, 1e-1, 1.4678e-1, 2.1544e-1]
drifts = [1e-4, 2.1544e-4, 4.6416e-4, 1e-3, 2.1544e-3, 4.6416e-3, 1e-2, 2.1544e-2, 4.6416e-2, 1e-1, 2.1544e-1, 4.6416e-1,]
#drifts = [4.64e-2, 7.06e-2, 1e-1, 1.42e-1, 2.15e-1,]


#drifts = [1e-4, 1e-3, 1e-2, 1e-1]
#drifts = [1e-4, 2.15e-4, 4.64e-4, 1e-3, 2.15e-3, 4.64e-3, 1e-2, 2.15e-2, 4.64e-2, 1e-1, 2.15e-1]
#drifts = [1e-2, 1e-1, 2.15e-1]



function line_to_plot(newsvendor_objective_value_and_order, ambiguity_radii, compute_weights, weight_parameters)

    μs = zeros(length(drifts))
    σs = zeros(length(drifts))

    for drift_index in eachindex(drifts)

        Random.seed!(42)
        drift_distribution = Uniform(-drifts[drift_index],drifts[drift_index])

        demand_sequences = [zeros(history_length) for _ in 1:repetitions]
        final_demand_probabilities = [zeros(1000) for _ in 1:repetitions]

        for repetition in 1:repetitions
            local demand_probability = initial_demand_probability

            for t in 1:history_length
                demand_sequences[repetition][t] = rand(Binomial(D, demand_probability))
                
                if t < history_length
                    demand_probability = min(max(demand_probability + rand(drift_distribution), 0), 1)

                else
                    for i in eachindex(final_demand_probabilities[repetition])
                        final_demand_probabilities[repetition][i] = min(max(demand_probability + rand(drift_distribution), 0), 1)
                
                    end
                end
            end
        end

        costs = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in 1:repetitions]
        doubling_count = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in 1:repetitions]

        precomputed_weights = [zeros(history_length) for weight_parameter_index in eachindex(weight_parameters)]

        #println("Precomputing weights...")
        Threads.@threads for weight_parameter_index in eachindex(weight_parameters)
            precomputed_weights[weight_parameter_index] = compute_weights(history_length, weight_parameters[weight_parameter_index])

        end

        #println("Parameter fitting...")
        Threads.@threads for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
            for repetition in 1:repetitions
                local weights = precomputed_weights[weight_parameter_index]
                local demand_samples = demand_sequences[repetition][1:history_length]

                local _, order, doubling_count[repetition][ambiguity_radius_index, weight_parameter_index] = newsvendor_objective_value_and_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights, 0)

                costs[repetition][ambiguity_radius_index, weight_parameter_index] = 
                    mean([expected_newsvendor_cost(order, final_demand_probabilities[repetition][i]) for i in eachindex(final_demand_probabilities[repetition])])

            end
        end

        display(compute_weights)

        digits = 4

        ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(costs)))
        minimal_costs = [costs[repetition][ambiguity_radius_index, weight_parameter_index] for repetition in 1:repetitions]
        μ = round(mean(minimal_costs), digits = digits)
        σ = round(sem(minimal_costs), digits = digits)
        print("Ex-post minimal average cost: $μ ± $σ, ")
        
        optimal_ambiguity_radius = round(ambiguity_radii[ambiguity_radius_index], digits = digits)
        optimal_weight_parameter = round(weight_parameters[weight_parameter_index], digits = digits)
        print("Optimal ambiguity radius: $optimal_ambiguity_radius, ")
        print("Weight parameter: $optimal_weight_parameter, ")

        optimal_doubling_count = round(mean([doubling_count[repetition][ambiguity_radius_index, weight_parameter_index] for repetition in 1:repetitions]), digits = digits)
        println("Doubling count: $optimal_doubling_count")

        μs[drift_index] = μ
        σs[drift_index] = σ

    end

    return μs, σs
end


LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))


#ε = [0; LinRange(1e-1,1e0,10); LinRange(2e0,1e1,9); LinRange(2e1,1e2,9);]
ε = [0; LinRange(1e-1,1e0,10); LinRange(2e0,1e1,9); LinRange(2e1,1e2,9); LinRange(2e2,1e3,9);]
s = unique(round.(Int, LogRange(1,history_length,30)))
α = [0; LogRange(1e-4,1e0,30)]
ρ╱ε = [0; LogRange(1e-4,1e0,30)]

#intersection_ε = [LinRange(1e-1,1e0,10); LinRange(2e0,1e1,9); LinRange(2e1,1e2,9);]
intersection_ε = [LinRange(1e-1,1e0,10); LinRange(2e0,1e1,9); LinRange(2e1,1e2,9); LinRange(2e2,1e3,9);]
intersection_ρ╱ε = [0; LogRange(1e-4,1e0,30)]
#intersection_ρ╱ε = [0; LogRange(1e-4,1e2,30)]

#line_to_plot(REMK_intersection_W2_newsvendor_objective_value_and_order, intersection_ε, REMK_intersection_weights, intersection_ρ╱ε)

    using Plots, Measures

    default() # Reset plot defaults.

    gr(size = (275+6+8,183+6).*sqrt(3))

    fontfamily = "Computer Modern"

    default(framestyle = :box,
            grid = true,
            #gridlinewidth = 1.0,
            gridalpha = 0.075,
            minorgrid = true,
            #minorgridlinewidth = 1.0, 
            minorgridalpha = 0.075,
            minorgridlinestyle = :dash,
            tick_direction = :in,
            xminorticks = 9, 
            yminorticks = 0,
            fontfamily = fontfamily,
            guidefont = Plots.font(fontfamily, pointsize = 12),
            legendfont = Plots.font(fontfamily, pointsize = 11),
            tickfont = Plots.font(fontfamily, pointsize = 10))

    plt = plot(xscale = :log10, #yscale = :log10,
                xlabel = "Binomial drift parameter, \$δ\$", 
                ylabel = "Ex-post optimal expected\ncost (relative to smoothing)",
                #title = "\$Ξ = [0,$D]\$, \$p_1\$\$ = $initial_demand_probability\$, \$T = $history_length\$",
                #title = "\$p_1\$\$ = $initial_demand_probability\$, \$s ≥ 0\$",
                #title = "\$p_1\$\$ = $initial_demand_probability\$, \$s > -∞\$",
                topmargin = 0pt,
                leftmargin = 6pt,
                bottommargin = 6pt,
                rightmargin = 0pt,
                legend = :bottomleft,
                )

    fillalpha = 0.1

    normalizer, normalizer_sems = line_to_plot(SO_newsvendor_objective_value_and_order, [0], smoothing_weights, α)
    expected_costs, sems = line_to_plot(SO_newsvendor_objective_value_and_order, [0], windowing_weights, [history_length])
    plot!(drifts, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
            color = palette(:tab10)[7],
            linestyle = :dashdot,
            markershape = :pentagon,
            markersize = 4,
            markerstrokewidth = 0,
            label = "SAA (\$ε=0\$)")

    expected_costs, sems = normalizer, normalizer_sems
    plot!(drifts, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
            color = palette(:tab10)[9],
            linestyle = :dot,
            linewidth = 1.2,
            markershape = :star4,
            markersize = 6,
            markerstrokewidth = 0,
            label = "Smoothing (\$ε=0\$)")

    expected_costs, sems = line_to_plot(REMK_intersection_W2_newsvendor_objective_value_and_order, intersection_ε, REMK_intersection_weights, intersection_ρ╱ε)
    plot!(drifts, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
            color = palette(:tab10)[1],
            linestyle = :solid,
            markershape = :circle,
            markersize = 4,
            markerstrokewidth = 0,
            label = "Intersection")

    expected_costs, sems = line_to_plot(W2_newsvendor_objective_value_and_order, ε, W2_weights, ρ╱ε)
    plot!(drifts, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
            color = palette(:tab10)[2],
            linestyle = :dash,
            markershape = :diamond,
            markersize = 4,
            markerstrokewidth = 0,
            label = "Weighted")

    xticks!([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    ylims!((0.7, 1.3))
    #xlims!((0.99999*drifts[1], 1.00001*drifts[end]))
    xlims!((0.99999*drifts[1], 1.00001*drifts[end-1]))

    display(plt)
    #savefig(plt, "figures/talk-ex-post-T=10.pdf")
    #savefig(plt, "figures/talk-ex-post-T=10.svg")
    
    #savefig(plt, "figures/talk-ex-post-T=70-alt.pdf")
    #savefig(plt, "figures/talk-ex-post-T=70-alt.svg")

    savefig(plt, "figures/talk-ex-post-T=70-alt-alt-alt.pdf")
    savefig(plt, "figures/talk-ex-post-T=70-alt-alt-alt.svg")





