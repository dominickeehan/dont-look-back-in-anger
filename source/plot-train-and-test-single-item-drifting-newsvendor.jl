using CSV, Statistics, StatsBase
using Plots, Measures

# Plots the data from the generate-train-and-test-single-item-drifting-newsvendor-data.jl script.

include("plot-sizing.jl")

costs = collect(CSV.File("data/single-item-drifting-newsvendor-costs.tsv", delim = '\t'))

drifts = sort(unique(row.drift for row in costs))
methods = ["SAA", "Smoothing", "Intersection", "Weighted"]

repetition_costs(method, drift) = [row.expected_next_period_cost for row in costs if row.method == method && row.drift == drift]

# Costs are drawn relative to the average cost of smoothing.
relative_average_costs(method) = [mean(repetition_costs(method, drift))/mean(repetition_costs("Smoothing", drift)) for drift in drifts]
relative_standard_errors(method) = [sem(repetition_costs(method, drift))/mean(repetition_costs("Smoothing", drift)) for drift in drifts]



default() # Reset plot settings to defaults.

# Widths, left to right: 38.06 for the y-axis title and tick labels, 240 for the plot box, and 38.06
# left blank, so that the plot box is centred on the page. Heights, top to bottom: 6.68 for the top
# tick label to stick out into, 160 for the plot box (a 3:2 aspect ratio), and 30.64 for the x-axis
# title and tick labels (see plot-sizing.jl).
# The sqrt(3) conversion means that (38.06 + 240 + 38.06, 6.68 + 160 + 30.64) is the size of the
# plot in points to specify in latex to correctly size the embedded fonts.
gr(size = (38.06 + 240 + 38.06, 6.68 + 160 + 30.64) .* sqrt(3))
fontfamily = "Computer Modern" # Close to Latin Modern.
default(framestyle = :box,
        grid = true,
        gridalpha = 0.075,
        minorgrid = true,
        minorgridalpha = 0.075,
        minorgridlinestyle = :dash,
        tick_direction = :in,
        xminorticks = 9,
        yminorticks = 0,
        fontfamily = fontfamily,
        guidefont = Plots.font(fontfamily, pointsize = 12), # Slight over emphasis on axis labels.
        legendfont = Plots.font(fontfamily, pointsize = 11),
        tickfont = Plots.font(fontfamily, pointsize = 10)) # Slight under emphasis on tick labels.

plt = plot(xscale = :log10,
           yscale = :log10,
           xlabel = "Drift parameter, \$δ\$",
           ylabel = "Expected cost (% difference)",
           legend = :topright)

colors = Dict("SAA" => palette(:tab10)[7], "Smoothing" => palette(:tab10)[9], "Intersection" => palette(:tab10)[1], "Weighted" => palette(:tab10)[2])
linestyles = Dict("SAA" => :dashdot, "Smoothing" => :dot, "Intersection" => :solid, "Weighted" => :dash)
linewidths = Dict("SAA" => 1.0, "Smoothing" => 1.2, "Intersection" => 1.0, "Weighted" => 1.0) # Slight over emphasis to make up for linestyles.
markershapes = Dict("SAA" => :pentagon, "Smoothing" => :star4, "Intersection" => :circle, "Weighted" => :diamond)
markersizes = Dict("SAA" => 4.0, "Smoothing" => 6.0, "Intersection" => 4.0, "Weighted" => 4.0)
fillalpha = 0.1 # Standard-error bands.

for method in methods
        plot!(drifts,
              relative_average_costs(method),
              ribbon = relative_standard_errors(method),
              fillalpha = fillalpha,
              color = colors[method],
              linestyle = linestyles[method],
              linewidth = linewidths[method],
              markershape = markershapes[method],
              markersize = markersizes[method],
              markerstrokewidth = 0.0,
              label = method)

end

xticks!([1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e0])
yticks!([0.8, 0.9, 1.0, 1.1, 1.2, 1.4], ["−20", "−10", "0", "+10", "+20", "+40"]) # Log-scaled cost ratios labelled as percentage differences.

# Tight axis limits.
xlims!((0.99999*drifts[begin], 1.00001*drifts[end]))
ylims!((0.79999, 1.50001))

place_plot_boxes!(plt, left = 38.06, right = 38.06, top = 6.68, bottom = 30.64)

display(plt)
savefig(plt, "figures/single-item-drifting-newsvendor-expected-cost.pdf")
