using CSV, Statistics, StatsBase
using Plots, Measures


const data_directory =
    joinpath(@__DIR__, "single-item-newsvendor-results")
const output_file = joinpath(
    @__DIR__,
    "output",
    "pdf",
    "average-train-and-test-single-item-next-period-expected-cost.pdf",
)
const methods = [
    "SAA",
    "Smoothing",
    "Intersection",
    "Weighted",
]


function result_csv_files(directory)
    files = filter(
        file -> occursin(
            r"^single-item-\d+\.csv$",
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
        Dict{Tuple{Float64,String},Vector{NamedTuple}}()

    for (file_index, file) in enumerate(files)
        println("Processing file $file_index of $(length(files))...")
        best_rows =
            Dict{Tuple{Float64,Int,String},NamedTuple}()

        for row in CSV.File(
            file;
            select = [
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

        for ((drift, _, method), result) in best_rows
            key = (drift, method)
            push!(
                get!(() -> NamedTuple[], selected_results, key),
                result,
            )
        end
    end

    drifts = sort!(unique(
        drift for (drift, _) in keys(selected_results)
    ))
    results = Dict{String,NamedTuple}()
    for method in methods
        selections = [
            selected_results[(drift, method)]
            for drift in drifts
        ]
        test_costs = [
            [selection.test_cost for selection in drift_selections]
            for drift_selections in selections
        ]
        results[method] = (
            average_costs = mean.(test_costs),
            standard_errors = sem.(test_costs),
            selected_results = selections,
        )
    end

    return (
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
    gr(size = (275 + 6 + 8 + 3, 183 + 6 + 10) .* sqrt(3))

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
        xlabel = "Binomial drift parameter, \$δ\$",
        ylabel =
            "Average train-and-test next-period\n" *
            "expected cost (relative to smoothing)",
        topmargin = 10.0pt,
        leftmargin = 6.0pt,
        bottommargin = 6.0pt,
        rightmargin = 3.0pt,
    )

    styles = Dict(
        "SAA" => (
            color = palette(:tab10)[7],
            linestyle = :dashdot,
            markershape = :pentagon,
            markersize = 4.0,
        ),
        "Smoothing" => (
            color = palette(:tab10)[9],
            linestyle = :dot,
            markershape = :star4,
            markersize = 6.0,
        ),
        "Intersection" => (
            color = palette(:tab10)[1],
            linestyle = :solid,
            markershape = :circle,
            markersize = 4.0,
        ),
        "Weighted" => (
            color = palette(:tab10)[2],
            linestyle = :dash,
            markershape = :diamond,
            markersize = 4.0,
        ),
    )

    normalizer =
        processed_results.results["Smoothing"].average_costs
    for method in processed_results.methods
        method_results = processed_results.results[method]
        style = styles[method]
        plot!(
            plt,
            processed_results.drifts,
            method_results.average_costs ./ normalizer;
            ribbon = method_results.standard_errors ./ normalizer,
            fillalpha = 0.1,
            color = style.color,
            linestyle = style.linestyle,
            linewidth = method == "Smoothing" ? 1.2 : 1.0,
            markershape = style.markershape,
            markersize = style.markersize,
            markerstrokewidth = 0.0,
            label = method,
        )
    end

    xlims!(plt, (
        0.99999 * first(processed_results.drifts),
        1.00001 * last(processed_results.drifts),
    ))
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
