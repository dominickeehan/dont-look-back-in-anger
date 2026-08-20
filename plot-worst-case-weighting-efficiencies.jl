# Worst-case efficiency of the triangular, windowing, and smoothing weighting
# schemes, measured against the optimal weights of weights.jl.
#
# For T historical observations, a Wasserstein order p, an ambiguity radius ε,
# a drift bound ρ, and a drift profile η, the concentration bound trades the
# part of the ambiguity radius which survives the drift of the data-generating
# distribution against the effective sample size, 1/Σₜwₜ². Wp_weights returns
# the weighting which maximises the resulting objective value
#
#     J(w) = (ε − ρ(Σₜ wₜ (T−t+1)^(pη))^(1/p))^(2p) / Σₜ wₜ² .
#
# The triangular, windowing, and smoothing schemes are one-parameter families
# of weightings, so each of them has an objective value too, namely the largest
# value of J attained over its horizon parameter. Their efficiency
#
#     E = (largest J over the horizon parameter) / (largest J over all
#         weightings) ∈ [0,1]
#
# is the fraction of the optimal objective value that the scheme retains, and
# this script plots the worst case of E over the parameter ranges
#
#     ε ∈ [0,1000],  ρ ∈ [0,1000],  p ∈ [1,5],  η ∈ [0,1],  T ∈ {10,100,1000},
#
# against the product pη, which is the exponent by which the drift of the
# data-generating distribution accumulates over the history. Each point of the
# plot is therefore the worst case over ε, over ρ, over the number of
# observations, and over the pairs (p,η) which multiply to that product.
#
# J is homogeneous of degree 2p in (ε,ρ), so E depends on ε and ρ only through
# the ratio ρ╱ε; and since Σₜ wₜ (T−t+1)^(pη) ≥ 1 for every weighting, ρ > ε
# leaves every scheme, optimal weights included, with a vacuous bound. Sweeping
# ε ∈ [0,1000] and ρ ∈ [0,1000] therefore reduces to sweeping ρ╱ε over [0,1],
# which is what is done below with ε set to one.

using Plots, Measures
using CSV
using ProgressBars

include("weights.jl")


const results_file =
    joinpath(@__DIR__, "worst-case-weighting-efficiency-data.tsv")
const figure_file =
    joinpath(@__DIR__, "figures", "worst-case-weighting-efficiencies.pdf")

# Set this to true to recompute the efficiencies. Otherwise the cached
# efficiencies are loaded.
refresh_efficiencies = !isfile(results_file)


LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))


# The triangular weighting scheme. It is defined here rather than in weights.jl
# because weights.jl holds the optimal weights that everything else is measured
# against.
#
# For a horizon parameter s ∈ [1,∞] the triangular scheme puts weight
# proportional to (s−(T−t+1))₊ on observation t: a linear ramp in time which
# reaches zero at the observation s periods old. The two endpoints are taken in
# the limit, so that s = 1 keeps only the most recent observation and s = ∞
# weights every observation equally. The scheme is parametrised below by the
# inverse horizon 1/s, which ranges over the bounded interval [0,1] and so
# admits a grid over the whole family.

function triangular_weights(T, inverse_horizon)

    if inverse_horizon == 0.0; weights = zeros(T); weights .= 1/T; return weights; end

    if inverse_horizon == 1.0; weights = zeros(T); weights[T] = 1.0; return weights; end

    s = 1/inverse_horizon

    weights = [max(s-(T-t+1), 0.0) for t in 1:T]
    weights = weights/sum(weights)

    return weights
end


# The powers (T−t+1)^(pη) which carry the drift into the objective value. They
# depend on the product pη alone, so they are formed once for each value of it.

drift_powers(T, drift_exponent) = [float(T-t+1)^drift_exponent for t in 1:T]


# The objective value maximised by Wp_weights, in logarithms and with ε set to
# one. At T = 1000 both the drift powers and the outer (⋅)^(2p) power are far
# outside the range of Float64, but the ratio of two objective values is not.
# Normalising by the total keeps the value invariant to the scaling of the
# weight vector, so that a weighting which sums to one only up to rounding is
# not reported as being infinitely better or worse than one which sums to one
# exactly.

function log_objective_value(total, square_total, drift_total, p, ρ╱ε)

    total > 0.0 || return -Inf

    drift = ρ╱ε * (drift_total/total)^(1/p)

    if drift >= 1.0; return -Inf; end # (The concentration bound is vacuous.)

    return 2*p*log(1-drift) - log(square_total/total^2)
end

function log_objective_value(weights, powers, p, ρ╱ε)

    total = 0.0
    square_total = 0.0
    drift_total = 0.0

    # Accumulated together so that the three sums are taken in the same order,
    # which keeps the drift of a weighting exactly ρ╱ε when pη = 0 rather than
    # a rounding either side of it. Every scheme here weights an observation no
    # more than a newer one, so the sums stop at the first zero.
    for t in lastindex(weights):-1:firstindex(weights)
        if weights[t] == 0.0; break; end

        total += weights[t]
        square_total += weights[t]^2
        drift_total += weights[t]*powers[t]
    end

    return log_objective_value(total, square_total, drift_total, p, ρ╱ε)
end


# The windowing scheme over every window size at once. A window of size L
# weights the L most recent observations equally, so the three sums it needs
# are L, L, and a running total of the drift powers over those L observations.
# Taking the weights as ones rather than as 1/L makes no difference, the
# objective value being invariant to the scaling of the weight vector. This is
# what makes searching all T window sizes cheaper than gridding them.

function best_windowing_log_objective_value(powers, p, ρ╱ε)

    best = -Inf
    drift_total = 0.0

    for window_size in eachindex(powers)
        drift_total += powers[lastindex(powers)-window_size+1]

        value = log_objective_value(float(window_size), float(window_size),
                                    drift_total, p, ρ╱ε)

        if value > best; best = value; end
    end

    return best
end


# The optimal objective value.
#
# The optimal weights, which Wp_weights solves for numerically, are a
# one-parameter family. Whichever drift Σₜ wₜ (T−t+1)^(pη) they attain, they
# must attain it with the largest possible effective sample size, so they
# minimise Σₜ wₜ² over the simplex subject to that drift; the first order
# conditions of that projection give
#
#     wₜ ∝ (1 − v(T−t+1)^(pη))₊ ,
#
# a linear ramp in the drift powers rather than in time, cut off at whichever
# observation is old enough for v(T−t+1)^(pη) to reach one. The whole family
# lies in v ∈ [0,1], from uniform weights (v = 0) to the most recent
# observation alone (v = 1), both of which are limits and are passed in below.
#
# Maximising over v is a one-dimensional search which stays well conditioned,
# where the interior point solve does not: at T = 1000 and pη = 5 the drift
# powers span fifteen orders of magnitude within a single constraint, and the
# solver returns badly suboptimal weights once pη ≳ 3 and ρ╱ε is not small.

function log_optimal_objective_value(powers, p, ρ╱ε, v)

    total = 0.0
    square_total = 0.0
    drift_total = 0.0

    # The ramp decreases in the age of the observation, so the sums stop at the
    # first observation which falls outside the cut off.
    for t in lastindex(powers):-1:firstindex(powers)
        weight = 1 - v*powers[t]

        if weight <= 0.0; break; end

        total += weight
        square_total += weight^2
        drift_total += weight*powers[t]
    end

    return log_objective_value(total, square_total, drift_total, p, ρ╱ε)
end

# The thresholds worth searching, in two stretches.
#
# Above 1/T^(pη) the cut off falls inside the history and the ramp is
# truncated, so the grid is taken in the cut off itself, z ∈ [1,T] with
# v = z^(−pη). Gridding v directly there would crowd every truncation into a
# sliver of its range whenever pη is small: at pη = 0.01 and T = 1000 the whole
# of the truncated ramps lies in v ∈ [0.93,1], which is a thirtieth of one
# decade.
#
# Below 1/T^(pη) the ramp reaches the whole history and only flattens onto
# uniform weights, which is a limit passed in separately. Cut offs up to T²
# cover the flattening where it still matters; three decades of v underneath
# carry the rest of the way.
const thresholds_per_decade = 250

function optimal_threshold_grid(powers, drift_exponent)

    T = length(powers)

    # Ascending in v, so that the bracket swept below is a bracket. The cut off
    # runs past T as well as up to it: past T the ramp is a trapezoid rather
    # than a truncated ramp, and the optimum sits there whenever ρ╱ε is close
    # to the ratio at which the bound turns vacuous.
    cut_offs = LogRange(float(T)^2, 1.0, ceil(Int, 2*thresholds_per_decade*log10(T)))
    truncated = [cut_off^(-drift_exponent) for cut_off in cut_offs]

    flattening = LogRange(1.0e-3/powers[begin], 1.0/powers[begin],
                          3*thresholds_per_decade) # (powers[begin] is T^(pη).)

    return [flattening; truncated]
end

function log_optimal_objective_value(powers, p, ρ╱ε, drift_exponent,
                                     uniform_weights, most_recent_weights)

    best = max(log_objective_value(uniform_weights, powers, p, ρ╱ε),
               log_objective_value(most_recent_weights, powers, p, ρ╱ε))

    v_values = optimal_threshold_grid(powers, drift_exponent)

    best_v_index = 0
    for v_index in eachindex(v_values)
        value = log_optimal_objective_value(powers, p, ρ╱ε, v_values[v_index])

        if value > best; best = value; best_v_index = v_index; end
    end

    # Refine on the bracket around the best point of the grid. The objective
    # value is not smooth in v, as the cut off moves past an observation
    # whenever v passes 1/(T−t+1)^(pη), so the bracket is swept rather than
    # sectioned.
    if best_v_index > 0
        lower = v_values[max(best_v_index-1, firstindex(v_values))]
        upper = v_values[min(best_v_index+1, lastindex(v_values))]

        for v in LinRange(lower, upper, 200)
            value = log_optimal_objective_value(powers, p, ρ╱ε, v)

            if value > best; best = value; end
        end
    end

    return best
end



# T. (100 is the experiments' history length; nothing moves once T is past
# about thirty, so three of them span the range.)
const horizons = [10, 100, 1000]

const maximum_order = 5.0
# ρ╱ε ∈ [0,1]. Finely resolved: the window size is an integer, so a scheme's
# efficiency scallops as ρ╱ε moves the best window from one integer to the
# next, and the worst case sits in one of those cusps. A coarse grid steps over
# the cusps and reports the schemes as better than they are, by as much as
# three points of a percent at sixty ratios.
const drift_ratio_values = [0.0; LogRange(1.0e-6, 1.0e0, 400)]

# The orders admissible at a drift exponent, p ∈ [max(1,pη), 5], as the drift
# profile η = pη/p must not exceed one. They are gridded inside that interval
# rather than on one fixed grid of p over [1,5], so that the endpoint η = 1,
# which is where the worst case often sits, is reached at every drift exponent
# instead of only at those which happen to land on a grid point of p. A fixed
# grid of p leaves a sawtooth in the worst case for exactly that reason.
const number_of_orders = 21

order_values(drift_exponent) =
    unique(LinRange(max(1.0, drift_exponent), maximum_order, number_of_orders))

# pη ∈ [0,5], as p ∈ [1,5] and η ∈ [0,1]. The turn towards pη = 0, where the
# drift stops depending on the weights and uniform weights become optimal, is
# resolved logarithmically: every scheme returns to being exactly efficient
# within the first tenth.
const drift_exponent_values =
    [0.0; LogRange(1.0e-3, 1.0e-1, 10); collect(0.2:0.1:maximum_order)]

# The horizon parameter ranges are those trained over in the experiments, see
# generate-train-and-test-multi-item-triptych-drifting-newsvendor-data.jl. The
# grids over them are refined so that an efficiency measures the limitation of
# the weighting scheme itself rather than the resolution of a tuning grid; set
# the refinement to 1 to recover the experiment grids.
const horizon_grid_refinement = 5
const number_of_horizon_values = 30 * horizon_grid_refinement

const methods = ["Triangular", "Windowing", "Smoothing"]


# The weightings of each scheme. They do not depend on p, ρ╱ε, or pη, so they
# are only constructed once for each number of observations.

function weight_vector_tables(T)

    # 1/s = 0 is uniform weights and 1/s = 1 is the most recent observation
    # alone; 1/s = 10^(-4) is a ramp flat to within a part in ten thousand.
    inverse_horizon_grid = [0.0; LogRange(1.0e-4, 1.0e0, number_of_horizon_values)]

    # α = 0 is uniform weights and α = 1 is the most recent observation alone.
    smoothing_parameter_grid = [0.0; LogRange(1.0e-4, 1.0e0, number_of_horizon_values)]

    # The windowing scheme needs no table: every window size is searched at
    # once by best_windowing_log_objective_value.
    return [[triangular_weights(T, inverse_horizon) for inverse_horizon in inverse_horizon_grid],
            Vector{Float64}[],
            [smoothing_weights(T, α) for α in smoothing_parameter_grid]]
end


const windowing_index = findfirst(==("Windowing"), methods)

function best_log_objective_value(method_index, tables, powers, p, ρ╱ε)

    if method_index == windowing_index
        return best_windowing_log_objective_value(powers, p, ρ╱ε)
    end

    return maximum(log_objective_value(weights, powers, p, ρ╱ε)
                   for weights in tables[method_index])
end


# The worst case, and the best case, over ρ╱ε of the efficiency of each scheme
# at one number of observations, one order, and one drift exponent. The work of
# one such combination is kept in its own function so that the threaded loop
# below does not share any of it. (A best case above one would say that a
# scheme beat the reference, and so that the search over thresholds was too
# coarse.)

function efficiencies_at(tables, T, p, drift_exponent, uniform_weights, most_recent_weights)

    powers = drift_powers(T, drift_exponent)

    worst_efficiencies = fill(Inf, length(methods))
    worst_drift_ratio_indices = fill(firstindex(drift_ratio_values), length(methods))
    best_efficiency = 0.0

    for drift_ratio_index in eachindex(drift_ratio_values)
        ρ╱ε = drift_ratio_values[drift_ratio_index]

        optimal_log_objective_value =
            log_optimal_objective_value(powers, p, ρ╱ε, drift_exponent,
                                        uniform_weights, most_recent_weights)

        for method_index in eachindex(methods)
            best = best_log_objective_value(method_index, tables, powers, p, ρ╱ε)

            # Where ρ ≥ ε the optimal bound is vacuous too, and so there is
            # nothing for a scheme to lose.
            efficiency =
                isfinite(optimal_log_objective_value) ?
                    exp(best - optimal_log_objective_value) :
                    (isfinite(best) ? Inf : 1.0)

            if efficiency < worst_efficiencies[method_index]
                worst_efficiencies[method_index] = efficiency
                worst_drift_ratio_indices[method_index] = drift_ratio_index
            end

            best_efficiency = max(best_efficiency, efficiency)
        end
    end

    return worst_efficiencies, worst_drift_ratio_indices, best_efficiency
end


# The worst case over both p and ρ╱ε, for every drift exponent, at one number
# of observations. Only the worst case is kept, along with where it is
# attained; the full grid is far too large to hold or to cache.

function compute_worst_case_efficiencies_at(T)

    tables = weight_vector_tables(T)
    uniform_weights = windowing_weights(T, T)
    most_recent_weights = windowing_weights(T, 1)

    grid = [(drift_exponent_index, order)
            for drift_exponent_index in eachindex(drift_exponent_values)
            for order in order_values(drift_exponent_values[drift_exponent_index])]

    grid_efficiencies = fill(NaN, length(methods), length(grid))
    grid_drift_ratios = fill(NaN, length(methods), length(grid))
    grid_best_efficiencies = fill(0.0, length(grid))

    progress_bar = ProgressBar(total = length(grid))

    Threads.@threads for grid_index in eachindex(grid)
        (exponent_index, order) = grid[grid_index]

        worst, worst_drift_ratio_indices, grid_best_efficiencies[grid_index] =
            efficiencies_at(tables,
                            T,
                            order,
                            drift_exponent_values[exponent_index],
                            uniform_weights,
                            most_recent_weights)

        for method_index in eachindex(methods)
            grid_efficiencies[method_index, grid_index] = worst[method_index]
            grid_drift_ratios[method_index, grid_index] =
                drift_ratio_values[worst_drift_ratio_indices[method_index]]
        end

        ProgressBars.update(progress_bar)
    end

    # Reduce over p, keeping the order and the ratio at which the worst case is
    # attained.
    efficiencies = fill(NaN, length(methods), length(drift_exponent_values))
    orders = fill(NaN, length(methods), length(drift_exponent_values))
    drift_ratios = fill(NaN, length(methods), length(drift_exponent_values))

    for grid_index in eachindex(grid)
        (exponent_index, order) = grid[grid_index]

        for method_index in eachindex(methods)
            efficiency = grid_efficiencies[method_index, grid_index]

            if isnan(efficiencies[method_index, exponent_index]) ||
                    efficiency < efficiencies[method_index, exponent_index]
                efficiencies[method_index, exponent_index] = efficiency
                orders[method_index, exponent_index] = order
                drift_ratios[method_index, exponent_index] =
                    grid_drift_ratios[method_index, grid_index]
            end
        end
    end

    overshoot = maximum(grid_best_efficiencies) - 1
    overshoot > 1.0e-4 && println("At T = $T a scheme beats the reference by up to " *
                                  "$(round(100*overshoot, digits = 4))%; " *
                                  "refine thresholds_per_decade.")

    return efficiencies, orders, drift_ratios
end


function compute_worst_case_efficiencies()

    efficiencies = fill(NaN, length(horizons), length(methods), length(drift_exponent_values))
    orders = fill(NaN, length(horizons), length(methods), length(drift_exponent_values))
    drift_ratios = fill(NaN, length(horizons), length(methods), length(drift_exponent_values))

    for horizon_index in eachindex(horizons)
        println("T = $(horizons[horizon_index]) on $(Threads.nthreads()) thread(s)...")

        (efficiencies[horizon_index, :, :],
         orders[horizon_index, :, :],
         drift_ratios[horizon_index, :, :]) =
            compute_worst_case_efficiencies_at(horizons[horizon_index])
    end

    return (efficiency = efficiencies, order = orders, drift_ratio = drift_ratios)
end


function save_worst_case_efficiencies(results, file = results_file)

    rows = NamedTuple[]
    for horizon_index in eachindex(horizons)
        for method_index in eachindex(methods)
            for drift_exponent_index in eachindex(drift_exponent_values)
                push!(rows,
                      (horizon = horizons[horizon_index],
                       method = methods[method_index],
                       drift_exponent = drift_exponent_values[drift_exponent_index],
                       worst_case_efficiency =
                           results.efficiency[horizon_index, method_index, drift_exponent_index],
                       worst_case_p =
                           results.order[horizon_index, method_index, drift_exponent_index],
                       worst_case_drift_ratio =
                           results.drift_ratio[horizon_index, method_index, drift_exponent_index]))
            end
        end
    end

    CSV.write(file, rows; delim = '\t')
    println("Saved worst-case efficiencies to $file.")

    return file
end


function load_worst_case_efficiencies(file = results_file)

    rows = collect(CSV.File(file; delim = '\t'))
    isempty(rows) && error("Efficiency data file is empty: $file")

    cached = Dict((Int(row.horizon), String(row.method), Float64(row.drift_exponent)) =>
                      (Float64(row.worst_case_efficiency),
                       Float64(row.worst_case_p),
                       Float64(row.worst_case_drift_ratio))
                  for row in rows)

    efficiencies = fill(NaN, length(horizons), length(methods), length(drift_exponent_values))
    orders = fill(NaN, length(horizons), length(methods), length(drift_exponent_values))
    drift_ratios = fill(NaN, length(horizons), length(methods), length(drift_exponent_values))

    for horizon_index in eachindex(horizons)
        for method_index in eachindex(methods)
            for drift_exponent_index in eachindex(drift_exponent_values)
                key = (horizons[horizon_index],
                       methods[method_index],
                       drift_exponent_values[drift_exponent_index])
                haskey(cached, key) ||
                    error("Missing cached efficiency for $key. " *
                          "Set refresh_efficiencies to true.")

                (efficiencies[horizon_index, method_index, drift_exponent_index],
                 orders[horizon_index, method_index, drift_exponent_index],
                 drift_ratios[horizon_index, method_index, drift_exponent_index]) = cached[key]
            end
        end
    end

    return (efficiency = efficiencies, order = orders, drift_ratio = drift_ratios)
end


function report_worst_case_efficiencies(results)

    println()
    for method_index in eachindex(methods)
        efficiencies = view(results.efficiency, method_index_slice(method_index)...)
        (horizon_index, drift_exponent_index) = Tuple(argmin(efficiencies))

        println(rpad(methods[method_index], 12) *
                "worst case efficiency " *
                "$(round(100*efficiencies[horizon_index, drift_exponent_index], digits = 2))% at " *
                "T = $(horizons[horizon_index]), " *
                "pη = $(round(drift_exponent_values[drift_exponent_index], digits = 2)), " *
                "p = $(results.order[horizon_index, method_index, drift_exponent_index]), " *
                "ρ╱ε = $(round(results.drift_ratio[horizon_index, method_index, drift_exponent_index], sigdigits = 3)).")
    end
    println()
end


# The efficiencies of one scheme, over every number of observations and every
# drift exponent.
method_index_slice(method_index) = (:, method_index, :)

# The worst case over the number of observations as well, which is what is
# plotted.
worst_case_over_horizons(results, method_index) =
    vec(minimum(view(results.efficiency, method_index_slice(method_index)...), dims = 1))


function plot_worst_case_efficiencies(results)

    default() # Reset to plot settings to defaults.

    # A 3:2 aspect ratio for the plot, excluding + 6 padding of the top, bottom,
    # and left margins. The sqrt(3) conversion means that (275 + 6 + 6, 183 + 6)
    # is the size of the plot in points to specify in latex to correctly size
    # the embedded fonts.
    gr(size = (275 + 6 + 6, 183 + 6) .* sqrt(3))
    set_plot_defaults()

    styles = method_styles()

    plt = plot(xlabel = "Drift exponent, \$pη\$",
               ylabel = "Worst-case efficiency (%)",
               xticks = (0:1:maximum_order),
               yticks = ([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"]),
               legend = :bottomleft,
               topmargin = 0pt,
               rightmargin = 6pt, # Padding.
               bottommargin = 6pt, # Padding.
               leftmargin = 12pt) # Padding. Consume more of the left margin to make up for the extra width of the y-axis label.

    for method_index in eachindex(methods)
        style = styles[method_index]

        plot!(plt,
              drift_exponent_values,
              worst_case_over_horizons(results, method_index),
              label = methods[method_index],
              color = style.color,
              linestyle = style.linestyle,
              linewidth = style.linewidth)
    end

    # Tight axis limits.
    xlims!(plt, (0.0, maximum_order))
    ylims!(plt, (0.0, 1.005)) # (With room for the frame.)

    return plt
end


function set_plot_defaults()

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
end


method_styles() = [(color = palette(:tab10)[1], linestyle = :solid, linewidth = 1.0),
                   (color = palette(:tab10)[7], linestyle = :dashdot, linewidth = 1.0),
                   (color = palette(:tab10)[9], linestyle = :dot, linewidth = 1.2)] # Slight over emphasis to make up for the linestyle.


results = if refresh_efficiencies
    computed_results = compute_worst_case_efficiencies()
    save_worst_case_efficiencies(computed_results)
    computed_results
else
    println("Loading worst-case efficiencies from $results_file.")
    load_worst_case_efficiencies()
end

report_worst_case_efficiencies(results)

plt = plot_worst_case_efficiencies(results)
display(plt)
savefig(plt, figure_file)
