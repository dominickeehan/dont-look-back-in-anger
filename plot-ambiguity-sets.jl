using Statistics, StatsBase
using Plots, Measures

include("weights.jl")

default() # Reset plot settings to defaults.

# A 3:2 aspect ratio for the plot, excluding + 6 padding of the bottom and left margins. 
# The sqrt(3) conversion means that (275 + 6, 183 + 6) is the size of the plot in points
# to specify in latex to correctly size the embedded fonts.
gr(size = (275 + 6, 183 + 6) .* sqrt(3))
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


samples = [1, -1, 2, 3]

p = 2

# In one dimension Gelbrich's bound states that for any distributions P and Q,
#
#       W2(P,Q)^2 >= (E[P]-E[Q])^2 + (std[P]-std[Q])^2.
#
# Thus, in the (mean, standard deviation)-plane, every Q within ε of P is 
# contained in a circle of radius ε centred around the coordinates of P.

mean_range = [-4, 8]
standard_deviation_range = [0, 7]

weighted_ε = 2.75
weighted_ρ = 1/3

scale_radii = 1.65 # The two approaches are scaled differently so that their sets are comparably sized.
intersection_ε = scale_radii * weighted_ε
intersection_ρ = weighted_ρ

P = [samples, Wp_power_law_drift_profile_weights(p, length(samples), weighted_ρ/weighted_ε, 1)]
weighted_ball = [mean(P[1], Weights(P[2])), std(P[1], Weights(P[2])), weighted_ε]

intersection_ball_radii = REMK_intersection_ball_radii(length(samples), intersection_ε, intersection_ρ/intersection_ε)
intersection_balls = [[samples[j], 0.0, intersection_ball_radii[j]] for j in eachindex(samples)]

function nonnegative_ball_coords(ball_parameters)

    x_centre, y_centre, radius = ball_parameters

    t = range(1.5*π, -0.5*π; length=1000)

    x_coords = x_centre .+ radius .* cos.(t)
    y_coords = y_centre .+ radius .* sin.(t)
    
    negative_y_coord_indices = y_coords .< 0
    y_coords[negative_y_coord_indices] .= 0

    return x_coords, y_coords

end

function nonnegative_intersected_balls_coords(ball_parameters)

    x_coords = range(maximum(x_centre - radius for (x_centre, _, radius) in ball_parameters),
                        minimum(x_centre + radius for (x_centre, _, radius) in ball_parameters); length=1000)

    y_coords = [minimum(sqrt(max(radius^2 - (x - x_centre)^2, 0)) for (x_centre, _, radius) in ball_parameters)
                    for x in x_coords]

    return x_coords, y_coords

end

# Ambiguity sets plot.

plt = plot(
        xlabel = "Mean", 
        ylabel = "Standard deviation",
        xlims = (mean_range[1],mean_range[end]),
        ylims = (standard_deviation_range[1],standard_deviation_range[end]),
        #legend = :horizontal,
        topmargin = 0pt, 
        rightmargin = 0pt,
        bottommargin = 6pt, # Padding.
        leftmargin = 6pt) # Padding.

linewidth = 1.5
alpha = 1
fillalpha = 0.075 # Visually nice.
total_fillalpha = 1-(1-fillalpha)^length(samples) # On 4 overlaid layers, this gives a total alpha of ≈ 0.268.

for ball_parameters in intersection_balls
        x_coords, y_coords = nonnegative_ball_coords(ball_parameters)
        plot!(x_coords,
                y_coords,
                color = palette(:tab10)[1],
                linewidth = 0,
                linestyle = :solid,
                alpha = 0,
                label = nothing,
                fill = (0, fillalpha, palette(:tab10)[1]))

end

x_coords, y_coords = nonnegative_intersected_balls_coords(intersection_balls)
plot!(x_coords,
        y_coords,
        color = palette(:tab10)[1],
        linewidth = linewidth,
        linestyle = :solid,
        alpha = alpha,
        label = nothing)

plot!([mean_range[1],mean_range[end]],
        [-1,-1],
        color = palette(:tab10)[1],
        linewidth = linewidth,
        linestyle = :solid,
        alpha = alpha,
        label = "Intersection",
        fill = (0, total_fillalpha, palette(:tab10)[1]))

x_coords, y_coords = nonnegative_ball_coords(weighted_ball)
plot!(x_coords,
        y_coords,
        color = palette(:tab10)[2],
        linewidth = linewidth,
        linestyle = :dash,
        alpha = alpha,
        label = "Weighted",
        fill = (0, total_fillalpha, palette(:tab10)[2]))

scatter!(samples,
        zeros(length(samples)), 
        markersize = 6.0,
        markershape = :utriangle,
        markercolor = :black,
        markerstrokecolor = :black,
        markerstrokewidth = 0,
        alpha = 1,
        labels = nothing)

for i in eachindex(samples); annotate!(samples[i], 0, text(" \$\\xi_$i\$", :black, :bottom, 12)); end

display(plt)
savefig(plt, "figures/ambiguity-sets.pdf")