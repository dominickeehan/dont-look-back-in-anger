using CSV, Statistics, StatsBase
using Plots, Measures

include("plot-sizing.jl")

# Draws the figure from the per-repetition costs written by
# generate-train-and-test-single-item-drifting-newsvendor-data.jl, so the plot
# setup can be adjusted without rerunning the experiment.

const costs_file = joinpath(
    @__DIR__,
    "single-item-drifting-newsvendor-costs.tsv",
)
const figure_file = joinpath(
    @__DIR__,
    "figures",
    "single-item-drifting-newsvendor-expected-cost.pdf",
)

# "Windowing" is also in the costs data, but is not drawn.
const methods = [
    "SAA",
    "Windowing",
    "Smoothing",
    "Intersection",
    "Weighted",
]


function load_single_item_costs(file = costs_file)
    grouped_costs = Dict{Tuple{Float64,String},Vector{Float64}}()
    for row in CSV.File(file; delim = '\t')
        push!(
            get!(
                () -> Float64[],
                grouped_costs,
                (Float64(row.drift), String(row.method)),
            ),
            Float64(row.expected_next_period_cost),
        )
    end
    isempty(grouped_costs) && error("Costs file is empty: $file")

    drifts = sort!(unique(drift for (drift, _) in keys(grouped_costs)))
    missing_cells = [
        (drift, method)
        for drift in drifts
        for method in methods
        if !haskey(grouped_costs, (drift, method))
    ]
    isempty(missing_cells) || error(
        "Missing single-item costs for: " *
        join(string.(missing_cells), ", "),
    )

    repetition_counts = unique(
        length(costs) for costs in values(grouped_costs)
    )
    println(
        "Loaded costs for $(length(drifts)) drifts, " *
        (
            length(repetition_counts) == 1 ?
            "$(only(repetition_counts)) repetitions per cell." :
            "$(minimum(repetition_counts)) to " *
            "$(maximum(repetition_counts)) repetitions per cell."
        ),
    )

    method_summary(method) = (
        average_costs = [
            mean(grouped_costs[(drift, method)]) for drift in drifts
        ],
        standard_errors = [
            sem(grouped_costs[(drift, method)]) for drift in drifts
        ],
    )
    results = (
        smoothing = method_summary("Smoothing"),
        SAA = method_summary("SAA"),
        windowing = method_summary("Windowing"),
        intersection = method_summary("Intersection"),
        weighted = method_summary("Weighted"),
    )
    return drifts, results
end

drifts, results = load_single_item_costs()


default() # Reset plot defaults.

# Widths, left to right: 38.06 for the y-axis title and tick labels, 240 for the plot box, and 38.06
# left blank, so that the plot box is centred on the page. Heights, top to bottom: 6.68 for
# the top tick label to stick out into, 160 for the plot box (a 3:2 aspect ratio), and 30.64 for the
# x-axis title and tick labels (see plot-sizing.jl).
# The sqrt(3) conversion means that (38.06 + 240 + 38.06, 6.68 + 160 + 30.64) is the size of the plot in points
# to specify in latex to correctly size the embedded fonts.
gr(size = (38.06 + 240 + 38.06, 6.68 + 160 + 30.64) .* sqrt(3))

fontfamily = "Computer Modern"
default(
    framestyle = :box,
    grid = true,
    gridalpha = 0.075,
    minorgrid = true,
    minorgridalpha = 0.075,
    minorgridlinestyle = :dash,
    tick_direction = :in,
    xminorticks = 9,
    yminorticks = 0,
    fontfamily = fontfamily,
    guidefont = Plots.font(fontfamily; pointsize = 12),
    legendfont = Plots.font(fontfamily; pointsize = 11),
    tickfont = Plots.font(fontfamily; pointsize = 10),
)

plt = plot(
    xscale = :log10,
    yscale = :log10,
    xlabel = "Drift parameter, \$δ\$",
    ylabel = "Expected cost (% difference)",
    legend = :topright,
)

fillalpha = 0.1
normalizer = results.smoothing.average_costs

plot!(
    plt,
    drifts,
    results.SAA.average_costs ./ normalizer;
    ribbon = results.SAA.standard_errors ./ normalizer,
    fillalpha = fillalpha,
    color = palette(:tab10)[7],
    linestyle = :dashdot,
    markershape = :pentagon,
    markersize = 4.0,
    markerstrokewidth = 0.0,
    label = "SAA",
)
plot!(
    plt,
    drifts,
    results.smoothing.average_costs ./ normalizer;
    ribbon = results.smoothing.standard_errors ./ normalizer,
    fillalpha = fillalpha,
    color = palette(:tab10)[9],
    linestyle = :dot,
    linewidth = 1.2,
    markershape = :star4,
    markersize = 6.0,
    markerstrokewidth = 0.0,
    label = "Smoothing",
)
plot!(
    plt,
    drifts,
    results.intersection.average_costs ./ normalizer;
    ribbon = results.intersection.standard_errors ./ normalizer,
    fillalpha = fillalpha,
    color = palette(:tab10)[1],
    linestyle = :solid,
    markershape = :circle,
    markersize = 4.0,
    markerstrokewidth = 0.0,
    label = "Intersection",
)
plot!(
    plt,
    drifts,
    results.weighted.average_costs ./ normalizer;
    ribbon = results.weighted.standard_errors ./ normalizer,
    fillalpha = fillalpha,
    color = palette(:tab10)[2],
    linestyle = :dash,
    markershape = :diamond,
    markersize = 4.0,
    markerstrokewidth = 0.0,
    label = "Weighted",
)
xticks!([1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e0])
xlims!((0.99999 * first(drifts), 1.00001 * last(drifts)))
# (Log-scaled cost ratios, labelled as percentage differences.)
yticks!(
    [0.8, 0.9, 1.0, 1.1, 1.2, 1.4],
    ["−20", "−10", "0", "+10", "+20", "+40"],
)
ylims!((0.79999, 1.50001))

place_plot_boxes!(plt, left = 38.06, right = 38.06, top = 6.68, bottom = 30.64)

display(plt)
savefig(plt, figure_file)
