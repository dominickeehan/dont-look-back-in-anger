using CSV, Statistics, StatsBase
using Plots, Measures


# Set this to true to process the real CSV data. Otherwise, use quick dummy
# data. Set refresh_processed_data to true when the source CSVs change;
# otherwise the small processed-data cache is loaded for a quick plot.
extract_data = true
refresh_processed_data = false

5

const data_directory =
    joinpath(@__DIR__, "triptych-drifting-newsvendor-results")
const processed_data_file =
    joinpath(@__DIR__, "multi-item-triptych-plot-data.tsv")
const methods = [
    "SAA",
    "Smoothing",
    "Intersection",
    "Weighted",
]
const item_counts_to_plot = 1:10
const drifts_to_plot = (0.01, 0.05, 0.1)


function result_csv_files(directory)
    files = filter(
        file -> occursin(
            r"^multi-item-triptych-\d+\.csv$",
            basename(file),
        ),
        readdir(directory; join = true),
    )
    sort!(
        files;
        by = file -> parse(
            Int,
            match(r"\d+", basename(file)).match,
        ),
    )
    return files
end


function process_train_and_test_data(directory = data_directory)
    files = result_csv_files(directory)
    isempty(files) && error("No triptych result CSV files found in $directory.")
    selected_results =
        Dict{Tuple{Int,Float64,String},Vector{NamedTuple}}()

    for (file_index, file) in enumerate(files)
        println("Processing file $file_index of $(length(files))...")
        best_rows =
            Dict{Tuple{Int,Float64,Int,String},NamedTuple}()

        for row in CSV.File(
            file;
            select = [
                :number_of_items,
                :drift,
                :repetition_index,
                :method,
                :ambiguity_radius,
                :weight_parameter,
                :average_training_cost,
                :objective_value,
                :expected_next_period_cost,
            ],
        )
            method = String(row.method)
            method in methods || continue
            number_of_items = Int(row.number_of_items)
            number_of_items in item_counts_to_plot || continue
            drift = Float64(row.drift)
            drift in drifts_to_plot || continue
            key = (
                number_of_items,
                drift,
                Int(row.repetition_index),
                method,
            )
            result = (
                training_cost = Float64(row.average_training_cost),
                objective_value = Float64(row.objective_value),
                test_cost = Float64(row.expected_next_period_cost),
                ambiguity_radius = Float64(row.ambiguity_radius),
                weight_parameter = Float64(row.weight_parameter),
            )

            if !haskey(best_rows, key) ||
                    result.training_cost <
                    best_rows[key].training_cost
                best_rows[key] = result
            end
        end

        for ((number_of_items, drift, _, method), result) in best_rows
            key = (number_of_items, drift, method)
            push!(
                get!(() -> NamedTuple[], selected_results, key),
                result,
            )
        end
    end

    item_counts = collect(item_counts_to_plot)
    drifts = collect(drifts_to_plot)
    missing_results = [
        (number_of_items, drift, method)
        for number_of_items in item_counts
        for drift in drifts
        for method in methods
        if !haskey(
            selected_results,
            (number_of_items, drift, method),
        )
    ]
    isempty(missing_results) || error(
        "Missing triptych results for: " *
        join(string.(missing_results), ", "),
    )

    results = Dict{Tuple{Int,String},NamedTuple}()
    for number_of_items in item_counts
        for method in methods
            selections = [
                selected_results[(number_of_items, drift, method)]
                for drift in drifts
            ]
            test_costs = [
                [selection.test_cost for selection in drift_selections]
                for drift_selections in selections
            ]
            results[(number_of_items, method)] = (
                average_costs = mean.(test_costs),
                standard_errors = sem.(test_costs),
                selected_results = selections,
            )
        end
    end

    return (
        item_counts = item_counts,
        drifts = drifts,
        methods = methods,
        results = results,
        number_of_files = length(files),
    )
end


function save_processed_train_and_test_data(
    processed_results,
    file = processed_data_file,
)
    rows = NamedTuple[]
    for number_of_items in processed_results.item_counts
        for (drift_index, drift) in enumerate(processed_results.drifts)
            for method in processed_results.methods
                method_result =
                    processed_results.results[(number_of_items, method)]
                push!(
                    rows,
                    (
                        number_of_items = number_of_items,
                        drift = drift,
                        method = method,
                        average_cost =
                            method_result.average_costs[drift_index],
                        standard_error =
                            method_result.standard_errors[drift_index],
                        number_of_source_files =
                            processed_results.number_of_files,
                    ),
                )
            end
        end
    end
    CSV.write(file, rows; delim = '\t')
    println("Saved processed plot data to $file.")
    return file
end


function load_processed_train_and_test_data(file = processed_data_file)
    rows = collect(CSV.File(file; delim = '\t'))
    isempty(rows) && error("Processed plot data file is empty: $file")

    cached_results = Dict(
        (
            Int(row.number_of_items),
            Float64(row.drift),
            String(row.method),
        ) => (
            average_cost = Float64(row.average_cost),
            standard_error = Float64(row.standard_error),
        )
        for row in rows
    )
    item_counts = collect(item_counts_to_plot)
    drifts = collect(drifts_to_plot)
    missing_results = [
        (number_of_items, drift, method)
        for number_of_items in item_counts
        for drift in drifts
        for method in methods
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
            results[(number_of_items, method)] = (
                average_costs = [
                    row.average_cost for row in method_rows
                ],
                standard_errors = [
                    row.standard_error for row in method_rows
                ],
                selected_results = [],
            )
        end
    end

    return (
        item_counts = item_counts,
        drifts = drifts,
        methods = methods,
        results = results,
        number_of_files = Int(first(rows).number_of_source_files),
    )
end


function dummy_train_and_test_data()
    levels = Dict(
        "SAA" => [1.06, 1.04, 1.02],
        "Smoothing" => ones(3),
        "Intersection" => [1.30, 1.20, 1.10],
        "Weighted" => [0.96, 0.93, 0.90],
    )
    standard_error = 0.012
    wiggle = 0.008
    results = Dict{Tuple{Int,String},NamedTuple}()

    for item_count in item_counts_to_plot
        for method in methods
            average_costs = levels[method] .+
                (
                    method == "Smoothing" ?
                    0.0 : wiggle * sin(item_count)
                )
            results[(item_count, method)] = (
                average_costs = average_costs,
                standard_errors = fill(
                    standard_error,
                    length(drifts_to_plot),
                ),
                selected_results = [],
            )
        end
    end

    return (
        item_counts = collect(item_counts_to_plot),
        drifts = collect(drifts_to_plot),
        methods = methods,
        results = results,
        number_of_files = 0,
    )
end


function plot_train_and_test_results(processed_results)
    default()
    a4_width_points = 210.0 / 25.4 * 72.0
    gr(size = (
        a4_width_points - 2.0 * 72.0,
        183 + 6 + 10,
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
            topmargin = 6.0pt,
            leftmargin = drift_index == firstindex(
                processed_results.drifts,
            ) ? 12.0pt : 1.0pt,
            bottommargin = 12.0pt,
            rightmargin = 0.0pt,
        )

        for method in processed_results.methods
            style = styles[method]
            relative_average_costs = [
                begin
                    method_result = processed_results.results[
                        (number_of_items, method)
                    ]
                    normalizer = processed_results.results[
                        (number_of_items, "Smoothing")
                    ].average_costs[drift_index]
                    method_result.average_costs[drift_index] /
                        normalizer
                end
                for number_of_items in processed_results.item_counts
            ]
            relative_standard_errors = [
                begin
                    method_result = processed_results.results[
                        (number_of_items, method)
                    ]
                    normalizer = processed_results.results[
                        (number_of_items, "Smoothing")
                    ].average_costs[drift_index]
                    method_result.standard_errors[drift_index] /
                        normalizer
                end
                for number_of_items in processed_results.item_counts
            ]
            # Plots uses markerstrokecolor and markerstrokewidth for error
            # bars, so draw them separately with method colours.
            scatter!(
                panel,
                processed_results.item_counts,
                relative_average_costs;
                yerror = relative_standard_errors,
                color = style.color,
                markershape = :none,
                markersize = 3.0,
                markerstrokecolor = style.color,
                markerstrokewidth = marker_outline_width,
                label = false,
            )
            plot!(
                panel,
                processed_results.item_counts,
                relative_average_costs;
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
    return plt
end


processed_results = if extract_data
    if refresh_processed_data || !isfile(processed_data_file)
        raw_processed_results = process_train_and_test_data()
        save_processed_train_and_test_data(raw_processed_results)
        raw_processed_results
    else
        println("Loading processed plot data from $processed_data_file.")
        load_processed_train_and_test_data()
    end
else
    dummy_train_and_test_data()
end
plt = plot_train_and_test_results(processed_results)
display(plt)
