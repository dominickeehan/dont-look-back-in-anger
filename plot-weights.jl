using Plots, Measures

include("weights.jl")

default() # Reset to plot settings to defaults.

# Single pane plots.

# A 3:2 aspect ratio for the plot, excluding + 6 padding of the top, bottom, and left margins. 
# The sqrt(3) conversion means that (275 + 6 + 6, 183 + 6) is the size of the plot in points
# to specify in latex to correctly size the embedded fonts.
gr(size = (275 + 6 + 6, 183 + 6) .* sqrt(3))
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
        xticks = ([0, 25, 50, 75, 100]),
        topmargin = 0pt, 
        rightmargin = 6pt, # Padding.
        bottommargin = 6pt, # Padding.
        leftmargin = 6pt) # Padding.

ε = 90
ρ = 1
T = 100
P = 1:5

linewidth = 1
colors = cgrad([palette(:tab10)[1], palette(:tab10)[2]], P[end])
linestyles = [:solid, :dash, :dashdot, :dashdotdot, :dot]
linewidths = LinRange(1,1.2,P[end]) # Slight over emphasis to make up for linestyles.
alpha = 0.117 # On 5 overlaid layers, this gives a total alpha of 1-(1-0.075)^5 ≈ 0.268, (see plot-ambiguity-sets.jl). 

for p in P
        plot!(1:T, 
                Wp_weights(p, T, (ρ/ε)/p, 1),
                label = "\$p=$p\$",
                color = colors[p],
                linewidth = linewidths[p],
                linestyle = linestyles[p],
                alpha = 1,
                fill = (0, alpha, colors[p]))

end

# Tight axis limits.
xlims!((-0,100)) 
yl = ylims(linear_drift_profile_plt)
ylims!((0,yl[2])) # (But keep natural upper limit.)

display(linear_drift_profile_plt)

savefig(linear_drift_profile_plt, "figures/linear-drift-profile-optimal-weights-for-p=1-to-5.pdf")


# Linear drift profile stop motion talk plots.
for q in 1:4
        plt = plot(
                title = "Linear drift profile \$(η=1)\$",
                xlabel = "Time index, \$t\$", 
                ylabel = "Optimal weight, \$w_t\$",
                xticks = ([0, 25, 50, 75, 100]),
                topmargin = 0pt, 
                rightmargin = 6pt,
                bottommargin = 6pt, 
                leftmargin = 6pt)

        for p in 1:q
                plot!(1:T, 
                        Wp_weights(p, T, (ρ/ε)/p, 1),
                        label = "\$p=$p\$",
                        color = colors[p],
                        linewidth = linewidths[p],
                        linestyle = linestyles[p],
                        alpha = 1,
                        fill = (0, alpha, colors[p]))

        end

        xlims!((-0,100))
        yl = ylims(plt)
        ylims!((0,yl[2]))

        display(plt)

        savefig(plt, "figures/linear-drift-profile-optimal-weights-for-p=1-to-$q.pdf")
end



# Square-root drift profile plot (η=1/2).

square_root_drift_profile_plt = plot(
        title = "Square-root drift profile \$(η=1/2)\$",
        xlabel = "Time index, \$t\$", 
        ylabel = "Optimal weight, \$w_t\$",
        xticks = ([0, 25, 50, 75, 100]),
        topmargin = 0pt, 
        rightmargin = 6pt, # Padding.
        bottommargin = 6pt, # Padding.
        leftmargin = 12pt) # Padding. Consume more of the left margin to make up for the extra width of the y-axis label.

ε = ε^(1/2)
ρ = ρ
T = T
P = P

linewidth = 1
colors = cgrad([palette(:tab10)[1], palette(:tab10)[2]], P[end])
linestyles = [:solid, :dash, :dashdot, :dashdotdot, :dot]
linewidths = LinRange(1,1.2,P[end]) # Slight over emphasis to make up for linestyles.
alpha = 0.117 # On 5 overlaid layers, this gives a total alpha of 1-(1-0.075)^5 ≈ 0.268, (see plot-ambiguity-sets.jl). 

for p in P
        plot!(1:T, 
                Wp_weights(p, T, (ρ/ε)/p, 1/2),
                label = "\$p=$p\$",
                color = colors[p],
                linewidth = linewidths[p],
                linestyle = linestyles[p],
                alpha = 1,
                fill = (0, alpha, colors[p]))

end

# Tight axis limits.
xlims!((-0,100)) 
yl = ylims(square_root_drift_profile_plt)
ylims!((0,yl[2])) # (But keep natural upper limit.)

display(square_root_drift_profile_plt)

savefig(square_root_drift_profile_plt, "figures/square-root-drift-profile-optimal-weights-for-p=1-to-5.pdf")



# Two pane plot.

# A 3:2 aspect ratio for the plot, excluding + 6 padding of the top, bottom, and left margins. 
# The sqrt(3) conversion means that (275 + 6 + 6, 183 + 6) is the size of the plot in points
# to specify in latex to correctly size the embedded fonts.

451
gr(size = (275 + 6 + 6, 183 + 6) .* sqrt(3))
gr(size = (275 + 6 + 6, 183 + 6) .* sqrt(3))


