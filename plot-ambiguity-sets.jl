using Statistics, StatsBase
using Plots, Measures

# Plots the Weighted and Intersection ambiguity sets in the (mean, standard deviation) plane.

include("weights.jl")
include("plot-sizing.jl")

default() # Reset plot settings to defaults.

# Widths, left to right: 26.03 for the y-axis title and tick labels, 240 for the plot box, and 26.03
# left blank, so that the plot box is centred on the page. Heights, top to bottom: 6.68 for the top
# tick label to stick out into, 160 for the plot box (a 3:2 aspect ratio), and 26.96 for the x-axis
# title and tick labels (see plot-sizing.jl).
# The sqrt(3) conversion means that (26.03 + 240 + 26.03, 6.68 + 160 + 26.96) is the size of the
# plot in points to specify in latex to correctly size the embedded fonts.
gr(size = (26.03 + 240 + 26.03, 6.68 + 160 + 26.96) .* sqrt(3))
fontfamily = "Computer Modern" # Close to Latin Modern.
default(framestyle = :box,
        grid = true,
        gridlinewidth = 0.5,
        gridalpha = 0.075,
        #minorgrid = true, # No minor grid for unscaled axes.
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

p = 2 # Wasserstein order.

mean_range = [-4, 8]
standard_deviation_range = [0, 7]

weighted_ε = 2.75
weighted_ρ = 1/3

scale_radii = 1.65 # Radii scaled so the two ambiguity sets are of comparable size.
intersection_ε = scale_radii * weighted_ε
intersection_ρ = weighted_ρ

# Balls of the two approaches in the (mean, standard deviation) plane. In one dimension Gelbrich's
# bound states that for any distributions P and Q
#
#       W2(P,Q)^2 >= (E[P]-E[Q])^2 + (std[P]-std[Q])^2.
#
# So every Q within ε of P lies in the disc of radius ε around the point of P.
P = [samples, Wp_power_law_drift_profile_weights(p, length(samples), weighted_ρ/weighted_ε, 1)]
weighted_ball = [mean(P[1], Weights(P[2])), std(P[1], Weights(P[2])), weighted_ε]

intersection_ball_radii = REMK_intersection_ball_radii(length(samples), intersection_ε, intersection_ρ/intersection_ε)
intersection_balls = [[samples[j], 0.0, intersection_ball_radii[j]] for j in eachindex(samples)]

# Circle around a ball with negative standard deviations set to zero.
function nonnegative_ball_coords(ball_parameters)

        x_centre, y_centre, radius = ball_parameters

        t = range(1.5*π, -0.5*π; length=1000)

        x_coords = x_centre .+ radius .* cos.(t)
        y_coords = y_centre .+ radius .* sin.(t)

        negative_y_coord_indices = y_coords .< 0
        y_coords[negative_y_coord_indices] .= 0

        return x_coords, y_coords

end

# Upper boundary of the intersection of the balls.
function nonnegative_intersected_balls_coords(ball_parameters)

        x_coords = range(maximum(x_centre - radius for (x_centre, _, radius) in ball_parameters),
                         minimum(x_centre + radius for (x_centre, _, radius) in ball_parameters); length=1000)

        y_coords = [minimum(sqrt(max(radius^2 - (x - x_centre)^2, 0)) for (x_centre, _, radius) in ball_parameters)
                    for x in x_coords]

        return x_coords, y_coords

end



# Ambiguity sets plot.

plt = plot(xlabel = "Mean",
           ylabel = "Standard deviation",
           xlims = (mean_range[begin], mean_range[end]),
           ylims = (standard_deviation_range[begin], standard_deviation_range[end]),
           #legend = :horizontal,
           )

linewidth = 1.5
alpha = 1
fill_alpha = 0.075 # Visually nice.
total_fill_alpha = 1-(1-fill_alpha)^length(samples) # On 4 overlaid layers, this gives the total alpha.

# Faint fill of each ball of the intersection approach.
for ball_parameters in intersection_balls
        x_coords, y_coords = nonnegative_ball_coords(ball_parameters)
        plot!(x_coords,
              y_coords,
              color = palette(:tab10)[1],
              linewidth = 0,
              linestyle = :solid,
              alpha = 0,
              label = nothing,
              fill = (0, fill_alpha, palette(:tab10)[1]))

end

# Boundary of the intersection.
x_coords, y_coords = nonnegative_intersected_balls_coords(intersection_balls)
plot!(x_coords,
      y_coords,
      color = palette(:tab10)[1],
      linewidth = linewidth,
      linestyle = :solid,
      alpha = alpha,
      label = nothing)

# Legend entry for the intersection drawn below the axis limits.
plot!([mean_range[begin], mean_range[end]],
      [-1, -1],
      color = palette(:tab10)[1],
      linewidth = linewidth,
      linestyle = :solid,
      alpha = alpha,
      label = "Intersection",
      fill = (0, total_fill_alpha, palette(:tab10)[1]))

# Ball of the weighted approach.
x_coords, y_coords = nonnegative_ball_coords(weighted_ball)
plot!(x_coords,
      y_coords,
      color = palette(:tab10)[2],
      linewidth = linewidth,
      linestyle = :dash,
      alpha = alpha,
      label = "Weighted",
      fill = (0, total_fill_alpha, palette(:tab10)[2]))

# Samples on the mean axis with their labels.
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

place_plot_boxes!(plt, left = 26.03, right = 26.03, top = 6.68, bottom = 26.96)

display(plt)
savefig(plt, "figures/ambiguity-sets.pdf")