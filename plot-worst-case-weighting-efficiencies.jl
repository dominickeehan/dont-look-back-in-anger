using Plots, Measures
using ProgressBars

include("weights.jl")

# Worst-case efficiency of the triangular, windowing, and smoothing weighting
# schemes, measured against the optimal weights of weights.jl. For T historical
# observations, a Wasserstein order p, a drift profile η, and a drift to radius
# ratio ρ╱ε, the concentration bound trades the part of the ambiguity radius
# which survives the drift of the data-generating distribution against the
# effective sample size, giving the objective value maximised by
# Wp_power_law_drift_profile_weights,
#
#       J(w) = (1-(Σₜ wₜ (T-t+1)^(ηp))^(1/p)ρ╱ε)^(2p) / Σₜ wₜ².
#
# Each scheme is a one-parameter family, so its efficiency is the largest J over
# its horizon parameter divided by the largest J over all weightings, and what
# is plotted is the worst case of that efficiency over
#
#       ρ╱ε ∈ [0,1], p ∈ [1,5], η ∈ [0,1], T ∈ {10,100,1000},
#
# against the drift exponent pη, by which the drift accumulates over the
# history. J is homogeneous of degree 2p in (ε,ρ), so ε and ρ enter only through
# their ratio, and as Σₜ wₜ (T-t+1)^(ηp) ≥ 1 the bound is vacuous for every
# weighting once ρ ≥ ε.


LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))


function triangular_weights(T, inverse_horizon)

    if inverse_horizon == 0.0; weights = zeros(T); weights .= 1/T; return weights; end

    weights = [max(1/inverse_horizon-(T-t+1), 0.0) for t in 1:T]

    if sum(weights) == 0.0; weights[T] = 1.0; return weights; end # (Only the most recent observation.)

    return weights/sum(weights)
end


# Whatever drift Σₜ wₜ (T-t+1)^(ηp) the optimal weights attain, they must attain
# it with the largest possible effective sample size, so they minimise Σₜ wₜ²
# subject to that drift and are a ramp in the drift powers,
#
#       wₜ ∝ (1-((T-t+1)/c)^(ηp))₊,
#
# cut off at the age c ∈ [1,∞), with uniform weights as the limit. Searching
# over c is well conditioned where the interior point solve of
# Wp_power_law_drift_profile_weights is not: at T = 1000 and ηp = 5 the drift
# powers span fifteen orders of magnitude within a single constraint, and the
# solver returns badly suboptimal weights once ηp ≳ 3.

function optimal_weights(T, drift_exponent, cut_off)

    weights = [max(1-((T-t+1)/cut_off)^drift_exponent, 0.0) for t in 1:T]

    if sum(weights) == 0.0; weights[T] = 1.0; return weights; end # (Only the most recent observation.)

    return weights/sum(weights)
end


# The drift powers (T-t+1)^(ηp) depend on the product ηp alone, so they are
# formed once for each value of it.

drift_powers(T, drift_exponent) = [float(T-t+1)^drift_exponent for t in 1:T]

function objective_value(w, powers, p, ρ╱ε)

    T = length(w)

    drift = (sum(w[t]*powers[t] for t in 1:T))^(1/p)*ρ╱ε

    if drift >= 1.0; return 0.0; end # (The bound is vacuous.)

    return (1/(sum(w[t]^2 for t in 1:T)))*((1-drift)^(2*p))
end

best_objective_value(weightings, powers, p, ρ╱ε) =
    maximum(objective_value(w, powers, p, ρ╱ε) for w in weightings)



const T_values = [10, 100, 1000] # (100 is the experiments' history length, and nothing moves once T is past about thirty.)

const maximum_order = 5.0

# pη ∈ [0,5], as p ∈ [1,5] and η ∈ [0,1]. Resolved logarithmically towards
# pη = 0, where the drift stops depending on the weights and every scheme is
# exactly efficient.
const drift_exponents = [0.0; LogRange(1.0e-3, 1.0e-1, 10); collect(0.2:0.1:maximum_order)]

# p ∈ [max(1,pη),5], as the drift profile η = pη/p must not exceed one. Gridded
# inside that interval rather than on one fixed grid of p over [1,5], so that
# η = 1, where the worst case often sits, is reached at every drift exponent.
const number_of_orders = 20
order_values(drift_exponent) = LinRange(max(1.0, drift_exponent), maximum_order, number_of_orders)

const number_of_parameter_values = 100

# ρ╱ε ∈ [0,1].
const drift_ratio_values = [0.0; LogRange(1.0e-6, 1.0e0, number_of_parameter_values)]

const methods = ["Triangular", "Windowing", "Smoothing"]

# The horizon parameter of each scheme, over the ranges trained over in the
# experiments (see generate-train-and-test-multi-item-triptych-drifting-
# newsvendor-data.jl), from uniform weights to the most recent observation
# alone. They do not depend on p, ρ╱ε, or pη, so they are built once for each T.
scheme_weightings(T) =
    [[triangular_weights(T, inverse_horizon) for inverse_horizon in [0.0; LogRange(1.0e-4, 1.0e0, number_of_parameter_values)]],
     [windowing_weights(T, window_size) for window_size in unique(round.(Int, LogRange(1, T, number_of_parameter_values)))],
     [smoothing_weights(T, α) for α in [0.0; LogRange(1.0e-4, 1.0e0, number_of_parameter_values)]]]



worst_efficiencies = fill(Inf, length(methods), length(drift_exponents))
best_efficiencies = fill(0.0, length(methods), length(drift_exponents))

for T in T_values

    println("T = $T on $(Threads.nthreads()) thread(s)...")

    weightings = scheme_weightings(T)
    cut_offs = LogRange(1.0, float(T)^2, number_of_parameter_values)

    progress_bar = ProgressBar(total = length(drift_exponents))

    Threads.@threads for i in eachindex(drift_exponents)

        powers = drift_powers(T, drift_exponents[i])
        optimal_weightings = [[windowing_weights(T, T)];
                              [optimal_weights(T, drift_exponents[i], cut_off) for cut_off in cut_offs]]

        for p in order_values(drift_exponents[i]), ρ╱ε in drift_ratio_values

            optimal = best_objective_value(optimal_weightings, powers, p, ρ╱ε)

            for m in eachindex(methods)
                best = best_objective_value(weightings[m], powers, p, ρ╱ε)

                # Where the optimal bound is vacuous there is nothing to lose.
                efficiency = optimal > 0.0 ? best/optimal : 1.0

                worst_efficiencies[m, i] = min(worst_efficiencies[m, i], efficiency)
                best_efficiencies[m, i] = max(best_efficiencies[m, i], efficiency)
            end
        end

        ProgressBars.update(progress_bar)
    end
end

println()
for m in eachindex(methods)
    i = argmin(worst_efficiencies[m, :])

    println(rpad(methods[m], 12) * "worst-case efficiency " *
            "$(round(100*worst_efficiencies[m, i], digits = 2))% at " *
            "pη = $(round(drift_exponents[i], digits = 2)).")
end

# A scheme beating the reference would say that the cut off grid is too coarse.
println("\nLargest efficiency $(round(100*maximum(best_efficiencies), digits = 4))%.\n")



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
        xticks = (0:1:maximum_order),
        yticks = ([0, 0.2, 0.4, 0.6, 0.8, 1], ["0", "20", "40", "60", "80", "100"]),
        legend = :bottomleft,
        topmargin = 0pt,
        rightmargin = 6pt, # Padding.
        bottommargin = 6pt, # Padding.
        leftmargin = 12pt) # Padding. Consume more of the left margin to make up for the extra width of the y-axis label.

colors = [palette(:tab10)[1], palette(:tab10)[7], palette(:tab10)[9]]
linestyles = [:solid, :dashdot, :dot]
linewidths = [1, 1, 1.2] # Slight over emphasis to make up for linestyles.

for m in eachindex(methods)
        plot!(drift_exponents,
                worst_efficiencies[m, :],
                label = methods[m],
                color = colors[m],
                linestyle = linestyles[m],
                linewidth = linewidths[m])

end

# Tight axis limits.
xlims!((0, maximum_order))
ylims!((0, 1.005)) # (With room for the frame.)

display(worst_case_efficiencies_plt)
savefig(worst_case_efficiencies_plt, "figures/worst-case-weighting-efficiencies.pdf")
