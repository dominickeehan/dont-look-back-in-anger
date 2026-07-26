using CSV, Statistics, StatsBase
using Plots, Measures


const data_directory =
    joinpath(@__DIR__, "multi-item-triptych-newsvendor-results")
const output_file = joinpath(
    @__DIR__,
    "output",
    "pdf",
    "average-train-and-test-multi-item-triptych-next-period-expected-cost.pdf",
)
const methods = [
    "Smoothing",
    "Intersection",
    "Weighted",
]
const drift_panel_labels = Dict(
    0.01 => "Small",
    0.1 => "Medium",
    0.5 => "Large",
)


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
            key = (
                Int(row.number_of_items),
                Float64(row.drift),
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

    item_counts = sort!(unique(
        number_of_items
        for (number_of_items, _, _) in keys(selected_results)
    ))
    drifts = sort!(unique(
        drift for (_, drift, _) in keys(selected_results)
    ))
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


function plot_train_and_test_results(
    processed_results;
    output_path = output_file,
)
    default()
    gr(size = (
        (210.0 / 25.4 - 2.0) * 72.0,
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
    )

    styles = Dict(
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

    fillalpha = 1.0 - 0.9^(1.0 / length(methods))
    ytick_values = collect(0.8:0.2:2.0)
    panel_plots = []
    for (drift_index, drift) in enumerate(processed_results.drifts)
        panel_yticks = if drift_index == firstindex(
            processed_results.drifts,
        )
            ytick_values
        else
            (ytick_values, fill("", length(ytick_values)))
        end
        panel = plot(
            xlabel = "Number of items",
            ylabel = drift_index == firstindex(
                processed_results.drifts,
            ) ?
                "Average train-and-test next-period\n" *
                "expected cost (relative to smoothing)" : "",
            title = "$(drift_panel_labels[drift]) drift " *
                "(\$δ = $drift\$)",
            xticks = processed_results.item_counts,
            xlims = (
                first(processed_results.item_counts) - 0.1,
                last(processed_results.item_counts) + 0.1,
            ),
            yticks = panel_yticks,
            ylims = (0.79999, 2.00001),
            legend = drift_index == lastindex(
                processed_results.drifts,
            ) ? :topright : false,
            topmargin = 5.0pt,
            leftmargin = drift_index == firstindex(
                processed_results.drifts,
            ) ? 14.0pt : 1.0pt,
            bottommargin = 13.0pt,
            rightmargin = 1.0pt,
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
                markerstrokewidth = 0.0,
                label = drift_index == lastindex(
                    processed_results.drifts,
                ) ? method : nothing,
            )
        end
        push!(panel_plots, panel)
    end

    plt = plot(
        panel_plots...;
        layout = (1, length(panel_plots)),
        link = :y,
    )
    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return plt
end


function process_and_plot_train_and_test_data(
    directory = data_directory;
    output_path = output_file,
)
    processed_results = process_train_and_test_data(directory)
    plt = plot_train_and_test_results(
        processed_results;
        output_path = output_path,
    )
    return (results = processed_results, plot = plt)
end


if abspath(PROGRAM_FILE) == @__FILE__
    output = process_and_plot_train_and_test_data()
    display(output.plot)
end
