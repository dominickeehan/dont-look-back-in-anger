using Plots, Measures

include("weights.jl")
include("plot-sizing.jl")

default() # Reset plot settings to defaults.

# Single pane plots.

# Widths, left to right: 41.22 for the y-axis title and tick labels, 240 for the plot box, and 41.22
# left blank, so that the plot box is centred on the page. Heights, top to bottom: 16.31 for
# the plot title, 160 for the plot box (a 3:2 aspect ratio), and 25.67 for the
# x-axis title and tick labels (see plot-sizing.jl).
# The sqrt(3) conversion means that (41.22 + 240 + 41.22, 16.31 + 160 + 25.67) is the size of the plot in points
# to specify in latex to correctly size the embedded fonts.
gr(size = (41.22 + 240 + 41.22, 16.31 + 160 + 25.67) .* sqrt(3))
fontfamily = "Computer Modern" # Close to Latin Modern.
default(framestyle = :box,
        grid = true,
        gridlinewidth = 0.5,
        gridalpha = 0.075,
        #minorgrid = true, # (No minor grid for unscaled axes.)
        #minorgridlinewidth = 1.0, 
        #minorgridalpha = 0.075,
        #minorgridlinestyle = :dash,
        tick_direction = :in,
        xminorticks = 0, 
        yminorticks = 0,
        fontfamily = fontfamily,
        titlefont = Plots.font(fontfamily, pointsize = 12), # Slight over emphasis on axis labels.
        guidefont = Plots.font(fontfamily, pointsize = 12), # Slight over emphasis on axis labels.
        legendfont = Plots.font(fontfamily, pointsize = 11),
        tickfont = Plots.font(fontfamily, pointsize = 10)) # Slight under emphasis on tick labels.



# Linear drift profile plot (η=1).

linear_drift_profile_plt = plot(
        title = "Linear drift profile \$(η=1)\$",
        xlabel = "Time index, \$t\$", 
        ylabel = "Optimal weight, \$w_t\$",
        xticks = ([0, 25, 50, 75, 100]))

ε = 90
ρ = 1
T = 100
P = 1:5

linewidth = 1
colors = cgrad([palette(:tab10)[1], palette(:tab10)[2]], P[end])
linestyles = [:solid, :dash, :dashdot, :dashdotdot, :dot]
linewidths = LinRange(1, 1.2, P[end]) # Slight over emphasis to make up for linestyles.
total_fill_alpha = 1 - (1-0.075)^8 # On 8 overlaid layers of alpha = 0.075 (see plot-ambiguity-sets.jl), this is the total alpha.
alpha = 1 - (1-total_fill_alpha)^(1/5) # On 5 overlaid layers, this gives a total alpha of total_fill_alpha.

for p in P
        plot!(1:T, 
                Wp_power_law_drift_profile_weights(p, T, (ρ/ε)/p, 1),
                label = "\$p=$p\$",
                color = colors[p],
                linewidth = linewidths[p],
                linestyle = linestyles[p],
                alpha = 1,
                fill = (0, alpha, colors[p]))

end

# Tight axis limits.
xlims!((-0, 100)) 
yl = ylims(linear_drift_profile_plt)
ylims!((0, yl[2])) # (But keep natural upper limit.)

place_plot_boxes!(linear_drift_profile_plt, left = 41.22, right = 41.22, top = 16.31, bottom = 25.67)

display(linear_drift_profile_plt)
savefig(linear_drift_profile_plt, "figures/linear-drift-profile-optimal-weights-for-p=1-to-5.pdf")


# Linear drift profile stop motion talk plots.
for q in 1:4
        plt = plot(
                title = "Linear drift profile \$(η=1)\$",
                xlabel = "Time index, \$t\$", 
                ylabel = "Optimal weight, \$w_t\$",
                xticks = ([0, 25, 50, 75, 100]))

        for p in 1:q
                plot!(1:T, 
                        Wp_power_law_drift_profile_weights(p, T, (ρ/ε)/p, 1),
                        label = "\$p=$p\$",
                        color = colors[p],
                        linewidth = linewidths[p],
                        linestyle = linestyles[p],
                        alpha = 1,
                        fill = (0, alpha, colors[p]))

        end

        xlims!((-0, 100))
        yl = ylims(plt)
        ylims!((0, yl[2]))

        place_plot_boxes!(plt, left = 41.22, right = 41.22, top = 16.31, bottom = 25.67)

        display(plt)
        savefig(plt, "figures/linear-drift-profile-optimal-weights-for-p=1-to-$q.pdf")
end



# Square-root drift profile plot (η=1/2).

square_root_drift_profile_plt = plot(
        title = "Square-root drift profile \$(η=1/2)\$",
        xlabel = "Time index, \$t\$", 
        ylabel = "Optimal weight, \$w_t\$",
        xticks = ([0, 25, 50, 75, 100]),
        size = (36.37 + 240 + 36.37, 16.16 + 160 + 25.52) .* sqrt(3)) # (Its tick labels differ from the linear drift profile plot's, so its own size.)

for p in P
        plot!(1:T, 
                Wp_power_law_drift_profile_weights(p, T, (ρ/ε^(1/2))/p, 1/2),
                label = "\$p=$p\$",
                color = colors[p],
                linewidth = linewidths[p],
                linestyle = linestyles[p],
                alpha = 1,
                fill = (0, alpha, colors[p]))

end

# Tight axis limits.
xlims!((-0, 100)) 
yl = ylims(square_root_drift_profile_plt)
ylims!((0, yl[2])) # (But keep natural upper limit.)

place_plot_boxes!(square_root_drift_profile_plt, left = 36.37, right = 36.37, top = 16.16, bottom = 25.52)

display(square_root_drift_profile_plt)
savefig(square_root_drift_profile_plt, "figures/square-root-drift-profile-optimal-weights-for-p=1-to-5.pdf")



# Two pane plot.

linear_drift_profile_two_pane_plt = plot(
        title = "Linear drift profile \$(η=1)\$",
        xlabel = "Time index, \$t\$",
        ylabel = "Optimal weight, \$w_t\$",
        xticks = ([0, 25, 50, 75, 100]),
        legend = :topleft)

square_root_drift_profile_two_pane_plt = plot(
        title = "Square-root drift profile \$(η=1/2)\$",
        xlabel = "Time index, \$t\$",
        xticks = ([0, 25, 50, 75, 100]),
        yformatter = _ -> "", # (Shared y-axis: tick numbers only on the left pane.)
        legend = false) # (Legend only on the left pane.)

for p in P
        plot!(linear_drift_profile_two_pane_plt,
                1:T,
                Wp_power_law_drift_profile_weights(p, T, (ρ/ε)/p, 1),
                label = "\$p=$p\$",
                color = colors[p],
                linewidth = linewidths[p],
                linestyle = linestyles[p],
                alpha = 1,
                fill = (0, alpha, colors[p]))

        plot!(square_root_drift_profile_two_pane_plt,
                1:T,
                Wp_power_law_drift_profile_weights(p, T, (ρ/ε^(1/2))/p, 1/2),
                label = "\$p=$p\$",
                color = colors[p],
                linewidth = linewidths[p],
                linestyle = linestyles[p],
                alpha = 1,
                fill = (0, alpha, colors[p]))
end

# Tight axis limits.
for plt in (linear_drift_profile_two_pane_plt, square_root_drift_profile_two_pane_plt); xlims!(plt, (0, 100)); end
# Shared y-axis limits (and hence tick positions) across both panes.
for plt in (linear_drift_profile_two_pane_plt, square_root_drift_profile_two_pane_plt)
        ylims!(plt, (0, ylims(square_root_drift_profile_two_pane_plt)[2]))
end

two_pane_drift_profile_plt = plot(
        linear_drift_profile_two_pane_plt,
        square_root_drift_profile_two_pane_plt,
        layout = (1, 2),
        size = (40.57 + 177.91 + 16 + 177.91 + 40.57, 18.27 + 160 + 27.62) .* sqrt(3)) # The width of an A4 piece of paper (excluding two 1-inch margins) is 452.97 points.

place_plot_boxes!(two_pane_drift_profile_plt, left = 40.57, gap = 16, right = 40.57, top = 18.27, bottom = 27.62)

display(two_pane_drift_profile_plt)
savefig(two_pane_drift_profile_plt, "figures/optimal-weights-for-p=1-to-5.pdf")
savefig(two_pane_drift_profile_plt, "figures/optimal-weights-for-p=1-to-5.svg") # (For the README.)




# Squished two pane plot.

squished_linear_drift_profile_two_pane_plt = plot(
        title = "Linear drift profile \$(η=1)\$",
        xlabel = "Time index, \$t\$",
        ylabel = "Optimal weight, \$w_t\$",
        xticks = ([0, 10, 20, 30, 40, 50]),
        legend = :topleft)

squished_square_root_drift_profile_two_pane_plt = plot(
        title = "Square-root drift profile \$(η=1/2)\$",
        xlabel = "Time index, \$t\$",
        xticks = ([0, 10, 20, 30, 40, 50]),
        yformatter = _ -> "", # (Shared y-axis: tick numbers only on the left pane.)
        legend = false) # (Legend only on the left pane.)

T = 50

for p in P
        plot!(squished_linear_drift_profile_two_pane_plt,
                1:T,
                Wp_power_law_drift_profile_weights(p, T, (ρ/ε)/p, 1),
                label = "\$p=$p\$",
                color = colors[p],
                linewidth = linewidths[p],
                linestyle = linestyles[p],
                alpha = 1,
                fill = (0, alpha, colors[p]))

        plot!(squished_square_root_drift_profile_two_pane_plt,
                1:T,
                Wp_power_law_drift_profile_weights(p, T, (ρ/ε^(1/2))/p, 1/2),
                label = "\$p=$p\$",
                color = colors[p],
                linewidth = linewidths[p],
                linestyle = linestyles[p],
                alpha = 1,
                fill = (0, alpha, colors[p]))
end

# Tight axis limits.
for plt in (squished_linear_drift_profile_two_pane_plt, squished_square_root_drift_profile_two_pane_plt); xlims!(plt, (0, T)); end
# Shared y-axis limits (and hence tick positions) across both panes.
for plt in (squished_linear_drift_profile_two_pane_plt, squished_square_root_drift_profile_two_pane_plt)
        ylims!(plt, (0, ylims(squished_square_root_drift_profile_two_pane_plt)[2]))
end

squished_two_pane_drift_profile_plt = plot(
        squished_linear_drift_profile_two_pane_plt,
        squished_square_root_drift_profile_two_pane_plt,
        layout = (1, 2),
        size = (40.57 + 177.91 + 16 + 177.91 + 40.57, 18.27 + 160 + 27.62) .* sqrt(3)) # The width of an A4 piece of paper (excluding two 1-inch margins) is 452.97 points.

place_plot_boxes!(squished_two_pane_drift_profile_plt, left = 40.57, gap = 16, right = 40.57, top = 18.27, bottom = 27.62)

display(squished_two_pane_drift_profile_plt)
savefig(squished_two_pane_drift_profile_plt, "figures/squished-optimal-weights-for-p=1-to-5.pdf")