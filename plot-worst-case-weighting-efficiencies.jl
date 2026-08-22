using Plots, Measures
using ProgressBars

include("weights.jl")

LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))

const T = 1000 # (The experiments use a history length of 100.)

const wasserstein_order_range = [1.0, 5.0]

const number_of_points = 200
const number_of_parameters_to_search = 300

# ηp terms appearing as (T-t+1)^(ηp) in the objective function.
const drift_term_exponents = unique([0.0; 
                                    sort(unique([LogRange(1.0e-9, 1.0e-0, Int(number_of_points/4)); 1 .- LogRange(1.0e-9, 1.0e-0, Int(number_of_points/4));])); 
                                    LogRange(1.0e-0, 5.0e-0, Int(number_of_points/2))])    

# ρ╱ε
const drift_ratios = [0; LogRange(1.0e-9, 1.0e0, 500)]

# p ∈ [max(1,pη),5], as the drift profile η = pη/p must not exceed one. Gridded
# inside that interval rather than on one fixed grid of p over [1,5], so that
# η = 1, where the worst case often sits, is reached at every drift exponent.
wasserstein_orders_above(drift_term_exponents) = LinRange(max(drift_term_exponents), wasserstein_order_range[end], number_of_points)

const schemes = ["Triangular", "Windowing", "Smoothing"]


# A ramp cut off somewhere between the smallest and the largest of the values it
# ramps in. The optimal weights ramp in the drift powers and the triangular
# scheme ramps in time, so the two families coincide at ηp = 1, where the drift
# powers are the times; they are swept over the same cut offs so that they still
# coincide once gridded.

horizons(start, stop, len) = unique([LogRange(start, stop, Int(len/2)); 1 ./ LinRange(0.0, 1/stop, Int(len/2))])

# The optimal weights are the reference every efficiency is measured against, so
# their cut offs are resolved four times as finely as the parameters of the
# windowing and smoothing schemes. The triangular scheme is swept over the same
# cut offs as they are, since at ηp = 1 it is the same family and would otherwise
# be measured against a finer grid of itself.



function triangular_weights(T, inverse_horizon)

    if inverse_horizon == 0.0; weights = zeros(T); weights .= 1/T; return weights; end
    if inverse_horizon == 1.0; weights = zeros(T); weights[T] = 1.0; return weights; end

    weights = [max(1.0/inverse_horizon-(T-t+1), 0.0) for t in 1:T]

    return weights/sum(weights)
end

# The weightings of each scheme, over the horizon parameter ranges trained over
# in the experiments. They do not depend on p, ρ╱ε, or pη, so they are built
# once, from uniform weights to the most recent observation alone.
const weighting_schemes =
    [[triangular_weights(T, 1.0/horizon) for horizon in horizons(1.0, float(T), number_of_parameters_to_search)], # (1/Inf = 0 gives uniform weights.)
     [windowing_weights(T, window_size) for window_size in unique(round.(Int, LogRange(1.0, float(T), number_of_parameters_to_search)))],
     [smoothing_weights(T, α) for α in [0.0; LogRange(1.0e-9, 1.0e0, number_of_parameters_to_search)]]]


5


     

# A weighting enters the objective value only through these two sums, and
# neither depends on p or ρ╱ε, so they are taken once for each weighting and the
# orders and ratios are then swept over them.

reusable_weighting_objective_terms(w, drift_terms) = (1/(sum(w[t]^2 for t in 1:T)), sum(w[t]*drift_terms[t] for t in 1:T))

function objective_value(reusable_terms, p, ρ╱ε)

    effective_sample_size, cumulative_drift = reusable_terms

    normalised_cumulative_drift = cumulative_drift^(1/p)*ρ╱ε

    if 1 - normalised_cumulative_drift <= 1.0e-9; return 0.0; end

    return (effective_sample_size)*((1-normalised_cumulative_drift)^(2*p))

end

# drift_terms[t] = (T-t+1)^(ηp)

function optimal_weights(drift_terms, horizon)

    if drift_terms[end]/horizon >= 1.0; weights = zeros(T); weights[T] = 1.0; return weights; end

    weights = [max(1-drift_term/horizon, 0.0) for drift_term in drift_terms]

    return weights/sum(weights)

end

# The cut offs worth sweeping, from the smallest drift power, where only the
# most recent observation survives, up to the largest, where the whole history
# is kept. Sweeping them logarithmically sweeps the age at which the ramp is cut
# off evenly; sweeping them linearly would place nearly every cut off within the
# last few observations whenever ηp is large.
#
# Above the largest drift power the whole history is kept and the ramp only
# flattens onto uniform weights. That stretch, which is where the optimum sits
# whenever ρ╱ε is small, is swept in 1/c instead, so that uniform weights are
# reached at 1/c = 0.



worst_efficiencies = fill(Inf, length(schemes), length(drift_term_exponents))
largest_efficiencies = fill(0.0, length(schemes), length(drift_term_exponents))

Threads.@threads for i in ProgressBar(eachindex(drift_term_exponents))

    drift_terms = [(T-t+1)^drift_term_exponents[i] for t in 1:T]

    optimal_sums = [reusable_weighting_objective_terms(optimal_weights(drift_terms, horizon), drift_terms) for horizon in horizons(drift_terms[end], drift_terms[1], number_of_points)]
    scheme_sums = [[reusable_weighting_objective_terms(w, drift_terms) for w in weightings] for weightings in weighting_schemes]

    for p in wasserstein_orders_above(drift_term_exponents[i]), ρ╱ε in drift_ratios

        optimal = maximum(objective_value(sums, p, ρ╱ε) for sums in optimal_sums)

        for s in eachindex(schemes)
            best = maximum(objective_value(sums, p, ρ╱ε) for sums in scheme_sums[s])

            # Where the optimal bound is vacuous there is nothing to lose.
            efficiency = optimal > 0.0 ? best/optimal : 1.0

            worst_efficiencies[s, i] = min(worst_efficiencies[s, i], efficiency)
            largest_efficiencies[s, i] = max(largest_efficiencies[s, i], efficiency)
        end
    end
end

println()
for s in eachindex(schemes)
    i = argmin(worst_efficiencies[s, :])

    println(rpad(schemes[s], 12) * "worst-case efficiency " *
            "$(round(100*worst_efficiencies[s, i], digits = 2))% at " *
            "pη = $(round(drift_term_exponents[i], digits = 2)).")
end

# A scheme beating the optimal weights measures how much the cut offs above miss
# them by.
println("\nLargest efficiency $(round(100*maximum(largest_efficiencies), digits = 4))%.\n")



default() # Reset plot settings to defaults.

# A 3:2 aspect ratio for the plot, excluding + 6 padding of the top, bottom, and left margins.
# The sqrt(3) conversion means that (275 + 6 + 6, 183 + 6) is the size of the plot in points
# to specify in latex to correctly size the embedded fonts.
# (275, 183) itself is just visually nice.
gr(size = (275 + 6 + 6, 183 + 6) .* sqrt(3))
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


worst_case_efficiencies_plt = plot(
        xlabel = "Drift exponent, \$pη\$",
        ylabel = "Worst-case efficiency (%)",
        xticks = (0:1:wasserstein_order_range[end], ["0", "1", "2", "3", "4", "5"]),
        yticks = ([0.5, 0.6, 0.7, 0.8, 0.9, 1], ["50", "60", "70", "80", "90", "100"]),
        legend = :bottomleft,
        topmargin = 0pt,
        rightmargin = 6pt, # Padding.
        bottommargin = 6pt, # Padding.
        leftmargin = 12pt) # Padding. Consume more of the left margin to make up for the extra width of the y-axis label.

colors = [palette(:tab10)[1], palette(:tab10)[7], palette(:tab10)[9]]
linestyles = [:solid, :dashdot, :dot]
linewidths = [1, 1, 1.2] # Slight over emphasis to make up for linestyles.
total_fill_alpha = 1-(1-0.075)^8 # On 8 overlaid layers of alpha = 0.075 (see plot-ambiguity-sets.jl), this is the total alpha.
alpha = 1-(1-total_fill_alpha)^(1/3) # On 3 overlaid layers, this gives a total alpha of total_fill_alpha.

for s in eachindex(schemes)
        plot!(drift_term_exponents,
                worst_efficiencies[s, :],
                label = schemes[s],
                color = colors[s],
                linestyle = linestyles[s],
                linewidth = linewidths[s],
                alpha = 1,
                fill = (0, alpha, colors[s]))

end

# Tight axis limits.
xlims!((0, wasserstein_order_range[end]))
ylims!((0.5, 1)) # (The lowest efficiency anywhere is 51.9%.)

display(worst_case_efficiencies_plt)
savefig(worst_case_efficiencies_plt, "figures/worst-case-weighting-efficiencies.pdf")
