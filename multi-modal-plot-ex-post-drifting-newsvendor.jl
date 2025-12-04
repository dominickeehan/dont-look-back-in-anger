using Random, Statistics, StatsBase, Distributions
using ProgressBars, IterTools

number_of_dimensions = 1
number_of_modes = 2
initial_demand_probabilities = [0.3 for _ in 1:number_of_modes]
numbers_of_consumers = [i*1000.0 for i in 1:number_of_modes]
number_of_consumers = max(numbers_of_consumers...)
cu = 4.0 # Per-unit underage cost.
co = 1.0 # Per-unit overage cost.
include("weights.jl")
include("newsvendor-optimizations.jl")

number_of_repetitions = 10 #300
history_length = 10 # 70

function expected_newsvendor_cost_with_binomial_demand(order, binomial_demand_probability, number_of_consumers)

    a = cdf(Binomial(number_of_consumers-1,binomial_demand_probability), order-1)
    b = cdf(Binomial(number_of_consumers,binomial_demand_probability), order)

    expected_underage_cost = cu * (number_of_consumers*binomial_demand_probability*(1-a) - order*(1-b))
    expected_overage_cost = co * (order*b - number_of_consumers*binomial_demand_probability*a)

    return expected_underage_cost + expected_overage_cost

end

drifts = [1e-3, 1e-2, 3e-2, 1e-1] # Same for each mode.

function line_to_plot(newsvendor_objective_value_and_order, ambiguity_radii, compute_weights, weight_parameters)

    average_costs = zeros(length(drifts))
    standard_deviations = zeros(length(drifts))

    for drift_index in eachindex(drifts)

        Random.seed!(42)
        drift_distribution = Uniform(-drifts[drift_index], drifts[drift_index])

        demand_sequences = [zeros(history_length) for _ in 1:number_of_repetitions]
        final_demand_probabilities = [[zeros(number_of_modes) for _ in 1:1000] for _ in 1:number_of_repetitions]

        for repetition_index in 1:number_of_repetitions
            local demand_probabilities = initial_demand_probabilities

            for t in 1:history_length
                demand_sequences[repetition_index][t] = 
                    rand(MixtureModel(Binomial, [(numbers_of_consumers[i], demand_probabilities[i]) for i in 1:number_of_modes]))
                
                if t < history_length
                    demand_probabilities = 
                        [min(max(demand_probabilities[i] + rand(drift_distribution[i]), 0), 1) for i in 1:number_of_modes]

                else
                    for i in eachindex(final_demand_probabilities[repetition_index])
                        final_demand_probabilities[repetition_index][i] = 
                            [min(max(demand_probabilities[i] + rand(drift_distribution[i]), 0), 1) for i in 1:number_of_modes]
                
                    end
                end
            end
        end

        costs = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in 1:number_of_repetitions]
        doubling_count = [zeros((length(ambiguity_radii),length(weight_parameters))) for _ in 1:number_of_repetitions]

        precomputed_weights = [zeros(history_length) for _ in eachindex(weight_parameters)]

        Threads.@threads for weight_parameter_index in eachindex(weight_parameters)
            precomputed_weights[weight_parameter_index] = compute_weights(history_length, weight_parameters[weight_parameter_index])

        end

        Threads.@threads for (ambiguity_radius_index, weight_parameter_index) in 
        #for (ambiguity_radius_index, weight_parameter_index) in 
            ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
            for repetition_index in 1:number_of_repetitions
                local weights = precomputed_weights[weight_parameter_index]
                local demand_samples = demand_sequences[repetition_index][1:history_length]

                local _, order, doubling_count[repetition_index][ambiguity_radius_index, weight_parameter_index] = 
                    newsvendor_objective_value_and_order(ambiguity_radii[ambiguity_radius_index], demand_samples, weights, 0)

                costs[repetition_index][ambiguity_radius_index, weight_parameter_index] = 
                    mean([sum(1/number_of_modes*expected_newsvendor_cost_with_binomial_demand(order, final_demand_probabilities[repetition_index][i][j], numbers_of_consumers[j]) for j in 1:number_of_modes) for i in eachindex(final_demand_probabilities[repetition_index])])

            end
        end

        display(compute_weights)

        digits = 4

        ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(costs)))
        minimal_costs = [costs[repetition_index][ambiguity_radius_index, weight_parameter_index] for repetition_index in 1:number_of_repetitions]
        average_cost = round(mean(minimal_costs), digits = digits)
        standard_deviation = round(sem(minimal_costs), digits = digits)
        print("Ex-post minimal average cost: $average_cost ± $standard_deviation, ")
        
        optimal_ambiguity_radius = round(ambiguity_radii[ambiguity_radius_index], digits = digits)
        optimal_weight_parameter = round(weight_parameters[weight_parameter_index], digits = digits)
        print("Optimal ambiguity radius: $optimal_ambiguity_radius, ")
        print("Weight parameter: $optimal_weight_parameter, ")

        optimal_doubling_count = round(mean([doubling_count[repetition_index][ambiguity_radius_index, weight_parameter_index] for repetition_index in 1:number_of_repetitions]), digits = digits)
        println("Doubling count: $optimal_doubling_count")

        average_costs[drift_index] = average_cost
        standard_deviations[drift_index] = standard_deviation

    end

    return average_costs, standard_deviations
end


LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))

discretisation = 10
ε = number_of_consumers*unique([0; LinRange(1e-3,1e-2,discretisation); LinRange(1e-2,1e-1,discretisation); LinRange(1e-1,1e-0,discretisation)])
s = unique(round.(Int, LogRange(1,history_length,3*discretisation)))
α = [0; LogRange(1e-4,1e0,3*discretisation)]
ρ╱ε = [0; LogRange(1e-4,1e0,3*discretisation)]
intersection_ε = number_of_consumers*unique([LinRange(1e-3,1e-2,discretisation); LinRange(1e-2,1e-1,discretisation); LinRange(1e-1,1e-0,discretisation)])
intersection_ρ╱ε = [0; LogRange(1e-4,1e0,3*discretisation)]


using Plots, Measures

default() # Reset plot defaults.

gr(size = (275+6+8+6,183+6).*sqrt(3))

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
            title = "modes \$= $number_of_modes\$, \$p_1 = $initial_demand_probabilities\$, \$Ξ = $numbers_of_consumers\$, \$T = $history_length\$",
            titlefontsize = 10,
            topmargin = 0pt,
            leftmargin = 6pt,
            bottommargin = 6pt,
            rightmargin = 6pt,
            legend = :bottomleft,
            )

fillalpha = 0.1

normalizer, normalizer_sems = line_to_plot(SO_newsvendor_objective_value_and_order, [0], smoothing_weights, α)
average_costs, sems = line_to_plot(SO_newsvendor_objective_value_and_order, [0], windowing_weights, [history_length])
plot!(drifts, average_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
        color = palette(:tab10)[7],
        linestyle = :dashdot,
        markershape = :pentagon,
        markersize = 4,
        markerstrokewidth = 0,
        label = "SAA (\$ε=0\$)")

average_costs, sems = normalizer, normalizer_sems
plot!(drifts, average_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
        color = palette(:tab10)[9],
        linestyle = :dot,
        linewidth = 1.2,
        markershape = :star4,
        markersize = 6,
        markerstrokewidth = 0,
        label = "Smoothing (\$ε=0\$)")

average_costs, sems = line_to_plot(REMK_intersection_W2_newsvendor_objective_value_and_order, intersection_ε, REMK_intersection_weights, intersection_ρ╱ε)
plot!(drifts, average_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
        color = palette(:tab10)[1],
        linestyle = :solid,
        markershape = :circle,
        markersize = 4,
        markerstrokewidth = 0,
        label = "Intersection")

average_costs, sems = line_to_plot(W2_newsvendor_objective_value_and_order, ε, W2_weights, ρ╱ε)
plot!(drifts, average_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
        color = palette(:tab10)[2],
        linestyle = :dash,
        markershape = :diamond,
        markersize = 4,
        markerstrokewidth = 0,
        label = "Weighted")


xticks!([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
ylims!((0.6, 1.4))
xlims!((0.99999*drifts[1], 1.00001*drifts[end]))

display(plt)

5


