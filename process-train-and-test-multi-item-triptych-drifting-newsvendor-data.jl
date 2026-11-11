using CSV, Printf, Statistics, StatsBase

# Reduces the raw HPC output (hundreds of multi-gigabyte CSV files) down to two
# small tables, so that the triptych plot can be redrawn without touching the
# raw data again:
#
#   * selected_runs_file -- one row per (number of items, drift, repetition,
#     method), holding the candidate that minimised the average training cost.
#     Everything the plot needs can be recomputed from this table in seconds.
#   * processed_data_file -- the plotted curves themselves, one row per (number
#     of items, drift, method).

const data_directory =
    joinpath(@__DIR__, "triptych-drifting-newsvendor-results")
const selected_runs_file =
    joinpath(@__DIR__, "multi-item-triptych-selected-runs.tsv")
const processed_data_file =
    joinpath(@__DIR__, "multi-item-triptych-plot-data.tsv")

# Curves are drawn relative to this method, so its per-repetition costs are
# also recorded as normalizers.
const normalizing_method = "Smoothing"
# Each job array index covers this many repetitions (see the .pbs script).
const number_of_repetitions_per_job = 10
# The .pbs script's job array runs indices 0 through number_of_jobs - 1, and
# every job's file must be present and complete for the summaries below.
const number_of_jobs = 300

const columns_to_read = [
    :number_of_items,
    :drift,
    :repetition_index,
    :method,
    :ambiguity_radius,
    :weight_parameter,
    :average_training_cost,
    :objective_value,
    :expected_next_period_cost,
]


function result_csv_files(directory = data_directory)
    files = filter(
        file -> occursin(
            r"^multi-item-triptych-\d+\.csv$",
            basename(file),
        ),
        readdir(directory; join = true),
    )
    sort!(files; by = job_number_of)
    missing_job_numbers =
        setdiff(0:(number_of_jobs - 1), job_number_of.(files))
    isempty(missing_job_numbers) || error(
        "Missing result files for job numbers " *
        join(missing_job_numbers, ", ") * ".",
    )
    return files
end


job_number_of(file) =
    parse(Int, match(r"\d+", basename(file)).match)


# Keeps, for every (number of items, drift, repetition, method) cell of one
# job's results, the candidate that minimises the average training cost. Ties
# go to the first row written, which matches the column-major order that
# argmin picks in the generating script.
function select_training_cost_minimizers(file)
    minimizers = Dict{Tuple{Int,Float64,Int,String},NamedTuple}()

    for row in CSV.File(file; select = columns_to_read)
        key = (
            Int(row.number_of_items),
            Float64(row.drift),
            Int(row.repetition_index),
            String(row.method),
        )
        average_training_cost = Float64(row.average_training_cost)

        if !haskey(minimizers, key) ||
                average_training_cost <
                minimizers[key].average_training_cost
            minimizers[key] = (
                ambiguity_radius = Float64(row.ambiguity_radius),
                weight_parameter = Float64(row.weight_parameter),
                average_training_cost = average_training_cost,
                objective_value = Float64(row.objective_value),
                expected_next_period_cost =
                    Float64(row.expected_next_period_cost),
            )
        end
    end

    return minimizers
end


function select_runs(files = result_csv_files())
    selected_runs = NamedTuple[]
    start_time = time()

    for (file_index, file) in enumerate(files)
        job_number = job_number_of(file)
        minimizers = select_training_cost_minimizers(file)

        for (
            (number_of_items, drift, repetition_index, method),
            minimizer,
        ) in minimizers
            push!(
                selected_runs,
                (
                    number_of_items = number_of_items,
                    drift = drift,
                    repetition_index =
                        number_of_repetitions_per_job * job_number +
                        repetition_index,
                    method = method,
                    ambiguity_radius = minimizer.ambiguity_radius,
                    weight_parameter = minimizer.weight_parameter,
                    average_training_cost =
                        minimizer.average_training_cost,
                    objective_value = minimizer.objective_value,
                    expected_next_period_cost =
                        minimizer.expected_next_period_cost,
                ),
            )
        end

        elapsed = time() - start_time
        @printf(
            "Processed %d of %d files (%.0f s elapsed, %.0f s remaining).\n",
            file_index,
            length(files),
            elapsed,
            elapsed * (length(files) - file_index) / file_index,
        )
        flush(stdout)
    end

    sort!(
        selected_runs;
        by = run -> (
            run.number_of_items,
            run.drift,
            run.method,
            run.repetition_index,
        ),
    )
    return selected_runs
end


# Every method sees the same demand sequences within a repetition, so cost
# ratios can be formed repetition by repetition. The paired relative statistics
# below use that; average_cost and standard_error do not, and are the
# quantities the current plot divides through by the normalizing method.
function summarize_selected_runs(selected_runs, number_of_files)
    normalizers = Dict(
        (run.number_of_items, run.drift, run.repetition_index) =>
            run.expected_next_period_cost
        for run in selected_runs if run.method == normalizing_method
    )

    grouped_runs = Dict{Tuple{Int,Float64,String},Vector{NamedTuple}}()
    for run in selected_runs
        push!(
            get!(
                () -> NamedTuple[],
                grouped_runs,
                (run.number_of_items, run.drift, run.method),
            ),
            run,
        )
    end

    curves = NamedTuple[]
    for (number_of_items, drift, method) in sort!(collect(keys(grouped_runs)))
        runs = grouped_runs[(number_of_items, drift, method)]
        test_costs = [run.expected_next_period_cost for run in runs]
        training_costs = [run.average_training_cost for run in runs]
        relative_test_costs = [
            run.expected_next_period_cost / normalizers[(
                run.number_of_items,
                run.drift,
                run.repetition_index,
            )]
            for run in runs
        ]

        push!(
            curves,
            (
                number_of_items = number_of_items,
                drift = drift,
                method = method,
                number_of_repetitions = length(test_costs),
                average_cost = mean(test_costs),
                standard_error = sem(test_costs),
                median_cost = median(test_costs),
                average_relative_cost = mean(relative_test_costs),
                relative_standard_error = sem(relative_test_costs),
                median_relative_cost = median(relative_test_costs),
                average_training_cost = mean(training_costs),
                number_of_source_files = number_of_files,
            ),
        )
    end

    return curves
end


# With the full job array present, every cell must hold the same number of
# repetitions; a shortfall means a result file was truncated.
function verify_even_coverage(curves)
    counts = unique(curve.number_of_repetitions for curve in curves)
    if length(counts) != 1
        shortfalls = join(
            (
                "items = $(curve.number_of_items), " *
                "drift = $(curve.drift), " *
                "method = $(curve.method): " *
                "$(curve.number_of_repetitions) repetitions"
                for curve in curves
                if curve.number_of_repetitions != maximum(counts)
            ),
            "; ",
        )
        error("Uneven repetition counts across cells: " * shortfalls * ".")
    end
    println("Every cell holds $(only(counts)) repetitions.")
    return nothing
end


files = result_csv_files()
println("Found $(length(files)) triptych result CSV files.")
selected_runs = select_runs(files)
CSV.write(selected_runs_file, selected_runs; delim = '\t')
println(
    "Saved $(length(selected_runs)) selected runs to $selected_runs_file.",
)

curves = summarize_selected_runs(selected_runs, length(files))
verify_even_coverage(curves)
CSV.write(processed_data_file, curves; delim = '\t')
println("Saved $(length(curves)) curve points to $processed_data_file.")
