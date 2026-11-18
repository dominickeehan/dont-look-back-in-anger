using CSV, Statistics, StatsBase
using ProgressBars

# Processes the data from the generate-train-and-test-multi-item-triptych-drifting-newsvendor-data.jl
# script for the plot-train-and-test-multi-item-triptych-drifting-newsvendor.jl script.

# HPC results with one file per job. The selected runs file keeps the run of each method with the
# smallest average training cost in every repetition for every number of items and drift. The plot
# data file averages these runs over the repetitions.
results_directory = "triptych-drifting-newsvendor-results"
selected_runs_file = "multi-item-triptych-selected-runs.tsv"
plot_data_file = "multi-item-triptych-plot-data.tsv"

number_of_jobs = 300 # Size of the .pbs job array.
number_of_repetitions_per_job = 10 # As in the generate script.


# Loop through the job files and keep the run of each method with the smallest average training cost
# in every repetition.
selected_runs = []

for job_number in ProgressBar(0:number_of_jobs-1)

        runs = CSV.File("$results_directory/multi-item-triptych-$job_number.csv",
                        select = [:number_of_items, :drift, :repetition_index, :method, :ambiguity_radius, :weight_parameter,
                                  :average_training_cost, :objective_value, :expected_next_period_cost])

        # Keep the first run with the smallest average training cost like argmin in train_and_test_order.
        best_runs = Dict()
        for run in runs

                key = (run.number_of_items, run.drift, run.repetition_index, run.method)

                if !haskey(best_runs, key) || run.average_training_cost < best_runs[key].average_training_cost; best_runs[key] = run; end

        end

        for run in values(best_runs)
                push!(selected_runs, (number_of_items = run.number_of_items,
                                      drift = run.drift,
                                      repetition_index = number_of_repetitions_per_job*job_number + run.repetition_index,
                                      method = String(run.method),
                                      ambiguity_radius = Float64(run.ambiguity_radius),
                                      weight_parameter = Float64(run.weight_parameter),
                                      average_training_cost = run.average_training_cost,
                                      objective_value = run.objective_value,
                                      expected_next_period_cost = run.expected_next_period_cost))

        end
end

# Save the selected runs in a fixed order.
sort!(selected_runs, by = run -> (run.number_of_items, run.drift, run.method, run.repetition_index))
CSV.write(selected_runs_file, selected_runs, delim = '\t')



# Smoothing costs as the comparator for the relative costs.
smoothing_costs = Dict((run.number_of_items, run.drift, run.repetition_index) => run.expected_next_period_cost
                       for run in selected_runs if run.method == "Smoothing")

# Loop through every number of items and drift and average the costs of each method over the repetitions.
plot_data = []

for (number_of_items, drift, method) in unique((run.number_of_items, run.drift, run.method) for run in selected_runs)

        runs = [run for run in selected_runs if (run.number_of_items, run.drift, run.method) == (number_of_items, drift, method)]

        costs = [run.expected_next_period_cost for run in runs]
        relative_costs = [run.expected_next_period_cost/smoothing_costs[(number_of_items, drift, run.repetition_index)] for run in runs]

        push!(plot_data, (number_of_items = number_of_items,
                          drift = drift,
                          method = method,
                          number_of_repetitions = length(runs),
                          average_cost = mean(costs),
                          standard_error = sem(costs),
                          median_cost = median(costs),
                          average_relative_cost = mean(relative_costs),
                          relative_standard_error = sem(relative_costs),
                          median_relative_cost = median(relative_costs),
                          average_training_cost = mean([run.average_training_cost for run in runs]),
                          number_of_source_files = number_of_jobs))

end

CSV.write(plot_data_file, plot_data, delim = '\t')
