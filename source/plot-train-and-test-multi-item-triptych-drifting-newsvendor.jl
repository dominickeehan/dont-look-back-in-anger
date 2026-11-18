using CSV
using Plots, Measures

# Plots the data from the process-train-and-test-multi-item-triptych-drifting-newsvendor-data.jl script.

include("plot-sizing.jl")

plot_data = collect(CSV.File("data/multi-item-triptych-plot-data.tsv", delim = '\t'))

item_counts = 1:10
drifts = [0.01, 0.05, 0.1]
methods = ["SAA", "Smoothing", "Intersection", "Weighted"]

plot_data_row(method, drift, item_count) = only(row for row in plot_data if row.method == method && row.drift == drift && row.number_of_items == item_count)

# Costs are drawn relative to the average cost of smoothing.
relative_average_costs(method, drift) = [plot_data_row(method, drift, item_count).average_cost/plot_data_row("Smoothing", drift, item_count).average_cost for item_count in item_counts]
relative_standard_errors(method, drift) = [plot_data_row(method, drift, item_count).standard_error/plot_data_row("Smoothing", drift, item_count).average_cost for item_count in item_counts]



default() # Reset plot settings to defaults.

# Widths, left to right: 42.16 for the y-axis title and tick labels, three 112.21 plot boxes 16
# apart, and 42.16 left blank, so that the plot boxes are centred on the page and fill the 452.97
# width of the text column. Heights, top to bottom: 20.02 for the plot titles, 160 for the plot
# boxes (as in every figure), and 29.37 for the x-axis titles and tick labels (see plot-sizing.jl).
# The sqrt(3) conversion means that (42.16 + 112.21 + 16 + 112.21 + 16 + 112.21 + 42.16, 20.02 + 160 + 29.37)
# is the size of the plot in points to specify in latex to correctly size the embedded fonts.
gr(size = (42.16 + 112.21 + 16 + 112.21 + 16 + 112.21 + 42.16, 20.02 + 160 + 29.37) .* sqrt(3))
fontfamily = "Computer Modern" # Close to Latin Modern.
default(framestyle = :box,
        grid = true,
        gridalpha = 0.075,
        minorgrid = true,
        minorgridalpha = 0.075,
        minorgridlinestyle = :dash,
        tick_direction = :in,
        xminorticks = 0,
        yminorticks = 0,
        fontfamily = fontfamily,
        titlefont = Plots.font(fontfamily, pointsize = 12), # Slight over emphasis on axis labels.
        guidefont = Plots.font(fontfamily, pointsize = 12), # Slight over emphasis on axis labels.
        legendfont = Plots.font(fontfamily, pointsize = 11),
        tickfont = Plots.font(fontfamily, pointsize = 10)) # Slight under emphasis on tick labels.

colors = Dict("SAA" => palette(:tab10)[7], "Smoothing" => palette(:tab10)[9], "Intersection" => palette(:tab10)[1], "Weighted" => palette(:tab10)[2])
linestyles = Dict("SAA" => :dashdot, "Smoothing" => :dot, "Intersection" => :solid, "Weighted" => :dash)
linewidths = Dict("SAA" => 1.0, "Smoothing" => 1.2, "Intersection" => 1.0, "Weighted" => 1.0) # Slight over emphasis to make up for linestyles.
markershapes = Dict("SAA" => :pentagon, "Smoothing" => :star4, "Intersection" => :circle, "Weighted" => :diamond)
markersizes = Dict("SAA" => 4.0, "Smoothing" => 6.0, "Intersection" => 4.0, "Weighted" => 4.0)
fillalpha = 0.1 # Standard-error bands, as in the single-item plot.

# Log-scaled cost ratios labelled as percentage differences.
ytick_values = [0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.8]
ytick_labels = ["−20", "−10", "0", "+10", "+20", "+40", "+80"]

panes = []

for drift in drifts

        plt = plot(title = "\$δ = $drift\$",
                   xlabel = "Number of items",
                   ylabel = drift == drifts[begin] ? "Expected cost (% difference)" : "",
                   xticks = item_counts,
                   yticks = (ytick_values, drift == drifts[begin] ? ytick_labels : fill("", length(ytick_values))), # Shared y-axis so only put tick numbers on the left pane.
                   yscale = :log10,
                   legend = drift == drifts[end] ? :right : false) # Legend only on the right pane.

        for method in methods
                plot!(plt,
                      item_counts,
                      relative_average_costs(method, drift),
                      ribbon = relative_standard_errors(method, drift),
                      fillalpha = fillalpha,
                      color = colors[method],
                      linestyle = linestyles[method],
                      linewidth = linewidths[method],
                      markershape = markershapes[method],
                      markersize = markersizes[method],
                      markerstrokecolor = :black,
                      markerstrokewidth = 1.0,
                      label = false)

        end

        # Tight axis limits.
        xlims!(plt, (0.9999*item_counts[begin], 1.0001*item_counts[end]))
        ylims!(plt, (0.9999*0.8, 1.00001*2.0))

        # Legend entries without the lines and bands. Drawn using hidden points outside of the axis limits.
        if drift == drifts[end]
                for method in methods
                        scatter!(plt,
                                 [-10.0, -11.0],
                                 [-10.0, -11.0],
                                 color = colors[method],
                                 alpha = 1.0,
                                 linewidth = 1.0,
                                 markershape = markershapes[method],
                                 markersize = 4.0,
                                 markerstrokewidth = 0.75,
                                 label = method)

                end

                xlims!(plt, (0.9999*item_counts[begin], 1.0001*item_counts[end]))
                ylims!(plt, (0.9999*0.8, 1.00001*2.0))
        end

        push!(panes, plt)

end

triptych_plt = plot(panes..., layout = (1, 3), link = :y)

place_plot_boxes!(triptych_plt, left = 42.16, gap = 16, right = 42.16, top = 20.02, bottom = 29.37)

display(triptych_plt)
savefig(triptych_plt, "figures/multi-item-triptych-drifting-newsvendor-expected-cost.pdf")
