using CSV
using Plots, Measures

include("plot-sizing.jl")

# Draws the triptych from the small processed table written by
# process-train-and-test-multi-item-triptych-drifting-newsvendor-data.jl, so
# the plot setup can be adjusted without rereading the raw HPC output.

const processed_data_file =
    joinpath(@__DIR__, "multi-item-triptych-plot-data.tsv")
const figure_file = joinpath(
    @__DIR__,
    "figures",
    "multi-item-triptych-drifting-newsvendor-expected-cost.pdf",
)

# "Windowing" is also in the processed data, but is not drawn.
const methods = [
    "SAA",
    "Smoothing",
    "Intersection",
    "Weighted",
]
const normalizing_method = "Smoothing"
const item_counts_to_plot = 1:10
const drifts_to_plot = (0.01, 0.05, 0.1)

# Every method sees the same demand sequences within a repetition, so the cost
# ratios can either be formed repetition by repetition and then averaged
# (paired, and so with tighter error bars), or by dividing the average costs
# through by the normalizing method's average cost, as they were originally
# drawn.
const use_paired_relative_costs = false


function load_processed_train_and_test_data(file = processed_data_file)
    rows = collect(CSV.File(file; delim = '\t'))
    isempty(rows) && error("Processed plot data file is empty: $file")

    cached_results = Dict(
        (
            Int(row.number_of_items),
            Float64(row.drift),
            String(row.method),
        ) => row
        for row in rows
    )
    item_counts = collect(item_counts_to_plot)
    drifts = collect(drifts_to_plot)
    missing_results = [
        (number_of_items, drift, method)
        for number_of_items in item_counts
        for drift in drifts
        for method in vcat(methods, normalizing_method)
        if !haskey(
            cached_results,
            (number_of_items, drift, method),
        )
    ]
    isempty(missing_results) || error(
        "Missing cached triptych results for: " *
        join(string.(missing_results), ", "),
    )

    results = Dict{Tuple{Int,String},NamedTuple}()
    for number_of_items in item_counts
        for method in methods
            method_rows = [
                cached_results[(number_of_items, drift, method)]
                for drift in drifts
            ]
            normalizer_rows = [
                cached_results[
                    (number_of_items, drift, normalizing_method)
                ]
                for drift in drifts
            ]
            results[(number_of_items, method)] =
                if use_paired_relative_costs
                    (
                        relative_average_costs = [
                            Float64(row.average_relative_cost)
                            for row in method_rows
                        ],
                        relative_standard_errors = [
                            Float64(row.relative_standard_error)
                            for row in method_rows
                        ],
                    )
                else
                    (
                        relative_average_costs = [
                            Float64(method_row.average_cost) /
                            Float64(normalizer_row.average_cost)
                            for (method_row, normalizer_row) in
                                zip(method_rows, normalizer_rows)
                        ],
                        relative_standard_errors = [
                            Float64(method_row.standard_error) /
                            Float64(normalizer_row.average_cost)
                            for (method_row, normalizer_row) in
                                zip(method_rows, normalizer_rows)
                        ],
                    )
                end
        end
    end

    return (
        item_counts = item_counts,
        drifts = drifts,
        methods = methods,
        results = results,
        repetition_counts = extrema(
            Int(cached_results[
                (number_of_items, drift, method)
            ].number_of_repetitions)
            for number_of_items in item_counts
            for drift in drifts
            for method in unique(vcat(methods, normalizing_method))
        ),
        number_of_files = Int(first(rows).number_of_source_files),
    )
end


function plot_train_and_test_results(processed_results)
    default()
    # Widths, left to right: 42.16 for the y-axis title and tick labels, three 112.21 plot
    # boxes 16 apart, and 42.16 left blank, so that the plot boxes are centred on the page
    # and fill the 452.97 width of the text column. Heights, top to bottom: 20.02 for the
    # plot titles, 160 for the plot boxes (as in every figure), and 29.37 for the x-axis
    # titles and tick labels (see plot-sizing.jl).
    # The sqrt(3) conversion means that
    # (42.16 + 112.21 + 16 + 112.21 + 16 + 112.21 + 42.16, 20.02 + 160 + 29.37) is the size of the plot
    # in points to specify in latex to correctly size the embedded fonts.
    gr(size = (
        42.16 + 112.21 + 16 + 112.21 + 16 + 112.21 + 42.16,
        20.02 + 160 + 29.37,
    ) .* sqrt(3))

    fontfamily = "Computer Modern"
    default(
        framestyle = :box,
        grid = true,
        gridalpha = 0.075,
        minorgrid = true,
        minorgridalpha = 0.075,
        minorgridlinestyle = :dash,
        tick_direction = :in,
        xminorticks = 0,
        yminorticks = 0,
        fontfamily = fontfamily,
        guidefont = Plots.font(fontfamily; pointsize = 12),
        legendfont = Plots.font(fontfamily; pointsize = 11),
        tickfont = Plots.font(fontfamily; pointsize = 10),
        titlefont = Plots.font(fontfamily; pointsize = 12),
    )

    styles = Dict(
        "SAA" => (
            color = palette(:tab10)[7],
            linestyle = :dashdot,
            linewidth = 1.0,
            markershape = :pentagon,
            markersize = 4.0,
        ),
        "Windowing" => (
            color = palette(:tab10)[5],
            linestyle = :dashdotdot,
            linewidth = 1.0,
            markershape = :utriangle,
            markersize = 4.0,
        ),
        "Smoothing" => (
            color = palette(:tab10)[9],
            linestyle = :dot,
            linewidth = 1.2,
            markershape = :star4,
            markersize = 6.0,
        ),
        "Intersection" => (
            color = palette(:tab10)[1],
            linestyle = :solid,
            linewidth = 1.0,
            markershape = :circle,
            markersize = 4.0,
        ),
        "Weighted" => (
            color = palette(:tab10)[2],
            linestyle = :dash,
            linewidth = 1.0,
            markershape = :diamond,
            markersize = 4.0,
        ),
    )
    marker_outline_color = :black
    marker_outline_width = 1.0
    fillalpha = 0.1 # (Standard-error bands, as in the single-item plot.)

    # (Log-scaled cost ratios, labelled as percentage differences.)
    ytick_values = [0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.8]
    ytick_labels = ["−20", "−10", "0", "+10", "+20", "+40", "+80"]
    panel_plots = []
    for (drift_index, drift) in enumerate(processed_results.drifts)
        panel_yticks = if drift_index == firstindex(
            processed_results.drifts,
        )
            (ytick_values, ytick_labels)
        else
            (ytick_values, fill("", length(ytick_values)))
        end
        panel = plot(
            xlabel = "Number of items",
            ylabel = drift_index == firstindex(
                processed_results.drifts,
            ) ? "Expected cost (% difference)" : "",
            title = "\$δ = $drift\$",
            xticks = processed_results.item_counts,
            xlims = (
                0.9999 * first(processed_results.item_counts),
                1.0001 * last(processed_results.item_counts),
            ),
            yticks = panel_yticks,
            yscale = :log10,
            ylims = (
                0.9999 * 0.8,
                1.00001 * 2.0,
            ),
            legend = drift_index == lastindex(
                processed_results.drifts,
            ) ? :right : false,
        )

        for method in processed_results.methods
            style = styles[method]
            relative_average_costs = [
                processed_results.results[
                    (number_of_items, method)
                ].relative_average_costs[drift_index]
                for number_of_items in processed_results.item_counts
            ]
            relative_standard_errors = [
                processed_results.results[
                    (number_of_items, method)
                ].relative_standard_errors[drift_index]
                for number_of_items in processed_results.item_counts
            ]
            plot!(
                panel,
                processed_results.item_counts,
                relative_average_costs;
                ribbon = relative_standard_errors,
                fillalpha = fillalpha,
                color = style.color,
                linestyle = style.linestyle,
                linewidth = style.linewidth,
                markershape = style.markershape,
                markersize = style.markersize,
                markerstrokecolor = marker_outline_color,
                markerstrokewidth = marker_outline_width,
                label = false,
            )
        end

        if drift_index == lastindex(processed_results.drifts)
            panel_xlims = xlims(panel)
            panel_ylims = ylims(panel)
            for method in processed_results.methods
                style = styles[method]
                scatter!(
                    panel,
                    [-10.0, -11.0],
                    [-10.0, -11.0];
                    color = style.color,
                    alpha = 1.0,
                    linewidth = 1.0,
                    markershape = style.markershape,
                    markersize = 4.0,
                    markerstrokewidth = 0.75,
                    label = "$method",
                )
            end
            xlims!(panel, panel_xlims)
            ylims!(panel, panel_ylims)
        end
        push!(panel_plots, panel)
    end

    plt = plot(
        panel_plots...;
        layout = (1, length(panel_plots)),
        link = :y,
    )
    place_plot_boxes!(plt, left = 42.16, gap = 16, right = 42.16, top = 20.02, bottom = 29.37)
    return plt
end


processed_results = load_processed_train_and_test_data()
fewest_repetitions, most_repetitions = processed_results.repetition_counts
println(
    "Loaded curves over $(processed_results.number_of_files) result files, " *
    (
        fewest_repetitions == most_repetitions ?
        "$most_repetitions repetitions per point." :
        "$fewest_repetitions to $most_repetitions repetitions per point."
    ),
)
plt = plot_train_and_test_results(processed_results)
display(plt)
savefig(plt, figure_file)
