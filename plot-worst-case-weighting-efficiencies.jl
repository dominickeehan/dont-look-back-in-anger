using Plots, Measures
using ProgressBars

include("weights.jl")

LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))

weighting_scheme_names = ["Triangular", "Windowing", "Smoothing"]
weighting_scheme_discretisation_fidelity = 1000

# Problem parameters to take the worst-case over.
T = 1000 # Larger T can only make the worst-case efficiencies worse, but they don't decrease much beyond T = 1000.
wasserstein_order_range = [1.0, 5.0]
drift_exponents = sort(unique([0.0; LogRange(1.0e-9, 1.0e-0, 100); 1 .- LogRange(1.0e-9, 1.0e-0, 100); LogRange(1.0e-0, 5.0e-0, 200)])) # (ηp terms appearing in (T-t+1)^(ηp).)
# For a given η'p', the p∈[1,5] such that there exists η∈[0,1] with ηp = η'p' is {p∈[1,5] : η=(η'p')/p ≤ 1} = [max(1,η'p'), 5].
wasserstein_orders(drift_exponent) = LogRange(max(wasserstein_order_range[begin], drift_exponent), wasserstein_order_range[end], 200)
drift_ratios = [0; LogRange(1.0e-9, 1.0e0, 1000)] # (ρ╱ε terms.)

horizons(start, stop, len) = sort(unique([LogRange(start, stop, Int(len/2)); 1 ./ LinRange(0.0, 1/stop, Int(len/2))])) # Go beyond stop to 1/0 = Inf, which gives uniform weights.

function triangular_weights(T, inverse_horizon)

    if inverse_horizon == 0.0; weights = zeros(T); weights .= 1/T; return weights; end
    if inverse_horizon == 1.0; weights = zeros(T); weights[T] = 1.0; return weights; end

    weights = [max(1.0/inverse_horizon-(T-t+1), 0.0) for t in 1:T]

    return weights/sum(weights)
end

weighting_schemes =
    [[triangular_weights(T, 1.0/horizon) for horizon in horizons(1, T, weighting_scheme_discretisation_fidelity)],
     [windowing_weights(T, window_size) for window_size in unique(round.(Int, LogRange(1, T, weighting_scheme_discretisation_fidelity)))],
     [smoothing_weights(T, α) for α in [0.0; LogRange(1.0e-9, 1.0e0, weighting_scheme_discretisation_fidelity)]]]
     
reusable_objective_terms(w, drift_terms) = (1/(sum(w[t]^2 for t in 1:T)), sum(w[t]*drift_terms[t] for t in 1:T))

function objective_value(reusable_terms, p, ρ╱ε)

    effective_sample_size, cumulative_drift = reusable_terms

    normalised_cumulative_drift = cumulative_drift^(1/p)*ρ╱ε

    if 1 - normalised_cumulative_drift <= 1.0e-9; return 0.0; end

    return (effective_sample_size)*((1-normalised_cumulative_drift)^(2*p))

end

function optimal_weights(drift_terms, horizon)

    if drift_terms[end]/horizon >= 1.0; weights = zeros(T); weights[T] = 1.0; return weights; end

    weights = [max(1-drift_term/horizon, 0.0) for drift_term in drift_terms]

    return weights/sum(weights)

end

worst_case_efficiencies = fill(Inf, length(weighting_schemes), length(drift_exponents))
best_case_efficiencies = fill(0.0, length(weighting_schemes), length(drift_exponents))

Threads.@threads for drift_exponent_index in ProgressBar(eachindex(drift_exponents))

    drift_terms = [(T-t+1)^drift_exponents[drift_exponent_index] for t in 1:T]

    optimal_weighting_scheme_terms = [reusable_objective_terms(optimal_weights(drift_terms, horizon), drift_terms) 
                                        for horizon in horizons(drift_terms[end], drift_terms[begin], weighting_scheme_discretisation_fidelity)]
    weighting_scheme_terms = [[reusable_objective_terms(w, drift_terms) for w in weightings] for weightings in weighting_schemes]

    for p in wasserstein_orders(drift_exponents[drift_exponent_index]), ρ╱ε in drift_ratios
        optimal_objective_value = maximum(objective_value(sums, p, ρ╱ε) for sums in optimal_weighting_scheme_terms)

        for scheme_index in eachindex(weighting_schemes)
            scheme_best_objective_value = maximum(objective_value(sums, p, ρ╱ε) for sums in weighting_scheme_terms[scheme_index])

            efficiency = optimal_objective_value > 0.0 ? scheme_best_objective_value/optimal_objective_value : 1.0

            worst_case_efficiencies[scheme_index, drift_exponent_index] = min(worst_case_efficiencies[scheme_index, drift_exponent_index], efficiency)
            best_case_efficiencies[scheme_index, drift_exponent_index] = max(best_case_efficiencies[scheme_index, drift_exponent_index], efficiency)

        end
    end
end

println()
for scheme_index in eachindex(weighting_schemes)
    i = argmin(worst_case_efficiencies[scheme_index, :])

    println(rpad(weighting_scheme_names[scheme_index], 12) * "worst-case efficiency " *
            "$(100*worst_case_efficiencies[scheme_index, i])% at " *
            "pη = $(drift_exponents[i]).")

end

# A scheme beating the optimal weights measures discretisation error.
println("\nLargest efficiency $(100*maximum(best_case_efficiencies))%.\n")


default() # Reset plot settings to defaults.

# A 3:2 aspect ratio for the plot, excluding + 6 padding of the top, bottom, and left margins.
# The sqrt(3) conversion means that (275 + 6 + 6, 183 + 6) is the size of the plot in points
# to specify in latex to correctly size the embedded fonts.
# (275, 183) itself is just visually nice.
gr(size = (275 + 6, 183 + 6) .* sqrt(3))
fontfamily = "Computer Modern" # Close to Latin Modern.
default(framestyle = :box,
        grid = true,
        gridlinewidth = 0.5,
        gridalpha = 0.075,
        tick_direction = :in,
        xminorticks = 0,
        yminorticks = 0,
        fontfamily = fontfamily,
        titlefont = Plots.font(fontfamily, pointsize = 12), # Slight over emphasis on axis labels.
        guidefont = Plots.font(fontfamily, pointsize = 12), # Slight over emphasis on axis labels.
        legendfont = Plots.font(fontfamily, pointsize = 11),
        tickfont = Plots.font(fontfamily, pointsize = 10)) # Slight under emphasis on tick labels.

plt = plot(
        xlabel = "Drift exponent, \$ηp\$",
        ylabel = "Worst-case efficiency (%)",
        xticks = (0:1:wasserstein_order_range[end], ["0", "1", "2", "3", "4", "5"]),
        yticks = ([0.5, 0.6, 0.7, 0.8, 0.9, 1], ["50", "60", "70", "80", "90", "100"]),
        legend = :bottomleft,
        topmargin = 0pt,
        rightmargin = 0pt,
        bottommargin = 6pt, # Padding.
        leftmargin = 6pt) # Padding.

colors = [palette(:tab10)[1], palette(:tab10)[7], palette(:tab10)[9]]
linestyles = [:solid, :dashdot, :dot]
linewidths = [1, 1, 1.2] # Slight over emphasis to make up for linestyles.
total_fill_alpha = 1 - (1-0.075)^8 # On 8 overlaid layers of alpha = 0.075 (see plot-ambiguity-sets.jl), this is the total alpha.
alpha = 1 - (1-total_fill_alpha)^(1/3) # On 3 overlaid layers, this gives a total alpha of total_fill_alpha.

for scheme_index in eachindex(weighting_schemes)
        plot!(drift_exponents,
                worst_case_efficiencies[scheme_index, :],
                label = weighting_scheme_names[scheme_index],
                color = colors[scheme_index],
                linestyle = linestyles[scheme_index],
                linewidth = linewidths[scheme_index],
                alpha = 1,
                fill = (0, alpha, colors[scheme_index]))

end

# Tight axis limits.
xlims!((0, wasserstein_order_range[end]))
ylims!((0.5, 1))

display(plt)
savefig(plt, "figures/worst-case-weighting-efficiencies.pdf")
