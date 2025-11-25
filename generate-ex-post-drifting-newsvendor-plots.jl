using Random, Statistics, StatsBase, Distributions
using ProgressBars, IterTools
using Plots, Measures

repetitions = 300
history_length = 70 #70

include("weights.jl")

initial_demand_probability = [0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5] # Binomial demand probability.
D = [100,1000,10000] # Number of consumers.
Cu = [1,2,3,4] # Per-unit underage cost.
Co = 1 # Per-unit overage cost.

job_number = parse(Int64, ENV["PBS_ARRAY_INDEX"])  # 0 to 119 (product of cardinality of above parameter sets).

# Compute indices
i_initial_demand_probability = job_number % length(initial_demand_probability)
job_number ÷= length(initial_demand_probability)
i_D = job_number % length(D)
job_number ÷= length(D)
i_Cu = job_number % length(Cu)
# Get values
initial_demand_probability = initial_demand_probability[i_initial_demand_probability + 1]
D = D[i_D + 1]
Cu = Cu[i_Cu + 1]
Co = Co

include("newsvendor-optimizations.jl")

function expected_newsvendor_cost(order, demand_probability)

    a = cdf(Binomial(D-1,demand_probability), order-1)
    b = cdf(Binomial(D,demand_probability), order)

    expected_underage_cost = Cu * (D*demand_probability*(1-a) - order*(1-b))
    expected_overage_cost = Co * (order*b - D*demand_probability*a)

    return expected_underage_cost + expected_overage_cost

end

#drifts = [1e-4, 2.1544e-4, 4.6416e-4, 1e-3, 2.1544e-3, 4.6416e-3, 1e-2, 2.1544e-2, 4.6416e-2, 1e-1] # Uniform drift.
drifts = [1e-4, 1e-3, 2.1544e-3, 4.6416e-3, 1e-2, 2.1544e-2, 4.6416e-2, 1e-1] # Uniform drift.


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
        #Threads.@threads for weight_parameter_index in eachindex(weight_parameters)
        for weight_parameter_index in eachindex(weight_parameters)
            precomputed_weights[weight_parameter_index] = compute_weights(history_length, weight_parameters[weight_parameter_index])

        end

        #println("Parameter fitting...")
        #Threads.@threads for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
        for (ambiguity_radius_index, weight_parameter_index) in ProgressBar(collect(IterTools.product(eachindex(ambiguity_radii), eachindex(weight_parameters))))
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


discretisation = 5
ε = D*unique([0; LinRange(1e-4,1e-3,discretisation); LinRange(1e-3,1e-2,discretisation); LinRange(1e-2,1e-1,discretisation)])
s = unique(round.(Int, LogRange(1,history_length,3*discretisation)))
α = [0; LogRange(1e-4,1e0,3*discretisation)]
ρ╱ε = [0; LogRange(1e-4,1e0,3*discretisation)]
intersection_ε = D*unique([LinRange(1e-4,1e-3,discretisation); LinRange(1e-3,1e-2,discretisation); LinRange(1e-2,1e-1,discretisation)])
intersection_ρ╱ε = [0; LogRange(1e-4,1e0,3*discretisation)]


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
            title = "\$Cu = $Cu\$, \$Ξ = [0,$D]\$, \$p_1\$\$ = $initial_demand_probability\$, \$T = $history_length\$",
            #title = "\$p_1\$\$ = $initial_demand_probability\$, \$s ≥ 0\$",
            #title = "\$p_1\$\$ = $initial_demand_probability\$, \$s > -∞\$",
            topmargin = 0pt,
            leftmargin = 6pt,
            bottommargin = 6pt,
            rightmargin = 6pt,
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
ylims!((0.5, 1.5))
xlims!((0.99999*drifts[1], 1.00001*drifts[end]))

display(plt)
savefig(plt, "ex-post-$job_number.pdf")



