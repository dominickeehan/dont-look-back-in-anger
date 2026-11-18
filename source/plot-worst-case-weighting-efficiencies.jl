using CSV
using Plots, Measures
using ProgressBars

# Plots the worst-case efficiencies of the triangular, windowing and smoothing weights.

include("weights.jl")
include("plot-sizing.jl")

# Computing the worst-case efficiencies takes a long time so they are saved to data_file and only
# recomputed if it is missing. Delete it to recompute them with
#
#     julia -t auto source/plot-worst-case-weighting-efficiencies.jl
#
# which takes about 15 minutes on 18 threads.
# In main.tex include the figure at width=\dimexpr 35.04pt + 240pt + 35.04pt\relax as it is drawn at
# the centred size (see plot-sizing.jl).
data_file = "data/worst-case-weighting-efficiency-data.tsv"

LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))

weighting_scheme_names = ["Triangular", "Windowing", "Smoothing"]
weighting_scheme_discretisation_fidelity = 1000

# Problem parameters to take the worst case over.
T = 1000 # Larger T can only lower the worst-case efficiencies but they barely change beyond 1000.
wasserstein_order_range = [1.0, 5.0]
drift_exponents = sort(unique([0.0; LogRange(1.0e-9, 1.0e-0, 100); 1 .- LogRange(1.0e-9, 1.0e-0, 100); LogRange(1.0e-0, 5.0e-0, 200)])) # Exponents ηp of the drift terms (T-t+1)^(ηp).
wasserstein_orders(drift_exponent) = LogRange(max(wasserstein_order_range[begin], drift_exponent), wasserstein_order_range[end], 200) # For a drift exponent ηp the orders p∈[1,5] with η = ηp/p ≤ 1 are [max(1,ηp), 5].
drift_ratios = [0; LogRange(1.0e-9, 1.0e0, 1000)] # Values of ρ╱ε.

horizons(start, stop, len) = sort(unique([LogRange(start, stop, Int(len/2)); 1 ./ LinRange(0.0, 1/stop, Int(len/2))])) # Goes beyond stop to 1/0 = Inf for uniform weights.

# Triangular weights that fall linearly to zero over the horizon.
function triangular_weights(T, inverse_horizon)

        if inverse_horizon == 0.0; weights = zeros(T); weights .= 1/T; return weights; end
        if inverse_horizon == 1.0; weights = zeros(T); weights[T] = 1.0; return weights; end

        weights = [max(1.0/inverse_horizon-(T-t+1), 0.0) for t in 1:T]

        return weights/sum(weights)
end



# Compute and save the worst-case efficiencies if data_file is missing.
if !isfile(data_file)

        # Weights of each scheme over its grid of parameters.
        weighting_schemes =
            [[triangular_weights(T, 1.0/horizon) for horizon in horizons(1, T, weighting_scheme_discretisation_fidelity)],
             [windowing_weights(T, window_size) for window_size in unique(round.(Int, LogRange(1, T, weighting_scheme_discretisation_fidelity)))],
             [smoothing_weights(T, α) for α in [0.0; LogRange(1.0e-9, 1.0e0, weighting_scheme_discretisation_fidelity)]]]

        # Effective sample size and cumulative drift of the weights w for reuse across p and ρ╱ε.
        reusable_objective_terms(w, drift_terms) = (1/(sum(w[t]^2 for t in 1:T)), sum(w[t]*drift_terms[t] for t in 1:T))

        # Objective value of weights from their reusable terms.
        function objective_value(reusable_terms, p, ρ╱ε)

                effective_sample_size, cumulative_drift = reusable_terms

                normalised_cumulative_drift = cumulative_drift^(1/p)*ρ╱ε

                if 1 - normalised_cumulative_drift <= 1.0e-9; return 0.0; end

                return (effective_sample_size)*((1-normalised_cumulative_drift)^(2*p))

        end

        # Value of ρ╱ε at which two weightings have equal objective values. It solves
        #
        #       ess_1^(1/(2p)) (1 - drift_1^(1/p) ρ╱ε) = ess_2^(1/(2p)) (1 - drift_2^(1/p) ρ╱ε).
        function equal_objective_drift_ratio(reusable_terms_1, reusable_terms_2, p)

                (effective_sample_size_1, cumulative_drift_1), (effective_sample_size_2, cumulative_drift_2) = reusable_terms_1, reusable_terms_2

                c_1, c_2 = effective_sample_size_1^(1/(2*p)), effective_sample_size_2^(1/(2*p))
                a_1, a_2 = cumulative_drift_1^(1/p), cumulative_drift_2^(1/p)

                return (c_2 - c_1)/(c_2*a_2 - c_1*a_1)

        end

        # Optimal weights in closed form for a given horizon. They fall to zero where the drift term reaches
        # the horizon and a line search over the horizon below finds the optimal weights without Ipopt.
        function optimal_weights(drift_terms, horizon)

                if drift_terms[end]/horizon >= 1.0; weights = zeros(T); weights[T] = 1.0; return weights; end

                weights = [max(1-drift_term/horizon, 0.0) for drift_term in drift_terms]

                return weights/sum(weights)

        end

        # Worst and best efficiency of each scheme for each drift exponent.
        worst_case_efficiencies = fill(Inf, length(weighting_schemes), length(drift_exponents))
        best_case_efficiencies = fill(0.0, length(weighting_schemes), length(drift_exponents))

        # Loop through every drift exponent, Wasserstein order and ρ╱ε and keep the worst and best
        # efficiency of each scheme.
        Threads.@threads for drift_exponent_index in ProgressBar(eachindex(drift_exponents))

                drift_terms = [(T-t+1)^drift_exponents[drift_exponent_index] for t in 1:T]

                # Reusable terms of the optimal weights for every horizon of the line search and of every scheme.
                optimal_weighting_scheme_terms = [reusable_objective_terms(optimal_weights(drift_terms, horizon), drift_terms)
                                                  for horizon in horizons(drift_terms[end], drift_terms[begin], weighting_scheme_discretisation_fidelity)]
                weighting_scheme_terms = [[reusable_objective_terms(w, drift_terms) for w in weightings] for weightings in weighting_schemes]
                windowing_terms = weighting_scheme_terms[findfirst(==("Windowing"), weighting_scheme_names)]

                for p in wasserstein_orders(drift_exponents[drift_exponent_index])

                        # Also check the ρ╱ε at which consecutive window sizes have equal objective values. The windowing
                        # efficiency has a sharp trough at each of these as the best window size switches there. The grid of
                        # drift_ratios alone misses the troughs by different amounts for different drift exponents which
                        # makes the worst-case windowing curve jagged.
                        switching_drift_ratios = filter(ρ╱ε -> 0.0 < ρ╱ε <= 1.0,
                                                        [equal_objective_drift_ratio(windowing_terms[k], windowing_terms[k+1], p) for k in 1:length(windowing_terms)-1])

                        for ρ╱ε in [drift_ratios; switching_drift_ratios]

                                # Optimal objective value from the line search over horizons.
                                optimal_objective_value = maximum(objective_value(sums, p, ρ╱ε) for sums in optimal_weighting_scheme_terms)

                                # Efficiency of each scheme against the optimal weights.
                                for scheme_index in eachindex(weighting_schemes)

                                        scheme_best_objective_value = maximum(objective_value(sums, p, ρ╱ε) for sums in weighting_scheme_terms[scheme_index])

                                        efficiency = optimal_objective_value > 0.0 ? scheme_best_objective_value/optimal_objective_value : 1.0

                                        worst_case_efficiencies[scheme_index, drift_exponent_index] = min(worst_case_efficiencies[scheme_index, drift_exponent_index], efficiency)
                                        best_case_efficiencies[scheme_index, drift_exponent_index] = max(best_case_efficiencies[scheme_index, drift_exponent_index], efficiency)

                                end
                        end
                end
        end

        # Print the worst-case efficiency of each scheme.
        println()
        for scheme_index in eachindex(weighting_schemes)

                worst_index = argmin(worst_case_efficiencies[scheme_index, :])

                println(rpad(weighting_scheme_names[scheme_index], 12) * "worst-case efficiency " *
                        "$(100*worst_case_efficiencies[scheme_index, worst_index])% at " *
                        "pη = $(drift_exponents[worst_index]).")

        end

        # Print the largest efficiency. Anything above 100% is discretisation error.
        println("\nLargest efficiency $(100*maximum(best_case_efficiencies))%.\n")

        # Save the worst-case and best-case efficiencies to data_file.
        CSV.write(data_file,
                  [(horizon = T,
                    method = weighting_scheme_names[scheme_index],
                    drift_exponent = drift_exponents[drift_exponent_index],
                    worst_case_efficiency = worst_case_efficiencies[scheme_index, drift_exponent_index],
                    best_case_efficiency = best_case_efficiencies[scheme_index, drift_exponent_index])
                   for scheme_index in eachindex(weighting_schemes) for drift_exponent_index in eachindex(drift_exponents)];
                  delim = '\t')
        println("Saved the worst-case efficiencies to $data_file.")
end



# Read the saved worst-case efficiencies and check they are on this script's grid of drift exponents.
efficiency_data = collect(CSV.File(data_file; delim = '\t'))
plotted_drift_exponents = [[row.drift_exponent for row in efficiency_data if row.method == name] for name in weighting_scheme_names]
plotted_worst_case_efficiencies = [[row.worst_case_efficiency for row in efficiency_data if row.method == name] for name in weighting_scheme_names]
all(exponents == drift_exponents for exponents in plotted_drift_exponents) ||
        error("$data_file was not computed on this script's grid of drift exponents, so delete it to recompute.")



default() # Reset plot settings to defaults.

# Widths, left to right: 35.04 for the y-axis title and tick labels, 240 for the plot box, and 35.04
# left blank, so that the plot box is centred on the page. Heights, top to bottom: 6.68 for the top
# tick label to stick out into, 160 for the plot box (a 3:2 aspect ratio), and 27.23 for the x-axis
# title and tick labels (see plot-sizing.jl).
# The sqrt(3) conversion means that (35.04 + 240 + 35.04, 6.68 + 160 + 27.23) is the size of the
# plot in points to specify in latex to correctly size the embedded fonts.
gr(size = (35.04 + 240 + 35.04, 6.68 + 160 + 27.23) .* sqrt(3))
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

plt = plot(xlabel = "Drift exponent, \$ηp\$",
           ylabel = "Worst-case efficiency (%)",
           xticks = (0:1:wasserstein_order_range[end], ["0", "1", "2", "3", "4", "5"]),
           yticks = ([0.5, 0.6, 0.7, 0.8, 0.9, 1], ["50", "60", "70", "80", "90", "100"]),
           legend = :bottomleft)

colors = [palette(:tab10)[1], palette(:tab10)[7], palette(:tab10)[9]]
linestyles = [:solid, :dashdot, :dot]
linewidths = [1, 1, 1.2] # Slight over emphasis to make up for linestyles.
total_fill_alpha = 1 - (1-0.075)^8 # On 8 overlaid layers of alpha = 0.075 (see plot-ambiguity-sets.jl), this is the total alpha.
alpha = 1 - (1-total_fill_alpha)^(1/3) # On 3 overlaid layers, this gives a total alpha of total_fill_alpha.

for scheme_index in eachindex(weighting_scheme_names)
        plot!(plotted_drift_exponents[scheme_index],
              plotted_worst_case_efficiencies[scheme_index],
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

place_plot_boxes!(plt, left = 35.04, right = 35.04, top = 6.68, bottom = 27.23)

display(plt)
savefig(plt, "figures/worst-case-weighting-efficiencies.pdf")
