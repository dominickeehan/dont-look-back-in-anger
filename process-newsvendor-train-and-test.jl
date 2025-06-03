# "C:\Program Files\7-Zip\7z.exe" e "C:\Users\dkee331\Documents\repositories\dont-look-back-in-anger\results\dominic(2).zip" -r -o"C:\Users\dkee331\Documents\repositories\dont-look-back-in-anger\results" *.csv

using CSV, Statistics, StatsBase, Plots

repetitions = 1
number_of_jobs = 1000

U = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01]

using ProgressBars

function extract_results(skipto, u_index)
    costs = Vector{Union{Missing, Float64}}(undef, number_of_jobs)
    parameter_1s = Vector{Union{Missing, Float64}}(undef, number_of_jobs)
    parameter_2s = Vector{Union{Missing, Float64}}(undef, number_of_jobs)
    
    Threads.@threads for job_number in ProgressBar(0:number_of_jobs-1)

        job_index = 999*(u_index-1)+job_number
        results_file = CSV.File("results/$job_index.csv", header = false, skipto = skipto*repetitions)
        try
            costs[job_number+1], parameter_1s[job_number+1], parameter_2s[job_number+1] = 
                [[row.Column2, row.Column3, row.Column4] for row in Iterators.take(results_file, repetitions)][1]
        catch
            costs[job_number+1], parameter_1s[job_number+1], parameter_2s[job_number+1] = [missing, missing, missing]
        end
    end

    return costs, parameter_1s, parameter_2s
end

u_index = 6

naive_W1_results = extract_results(1, u_index)
windowing_W1_results = extract_results(2, u_index)
smoothing_W1_results = extract_results(3, u_index)
concentration_W1_results = extract_results(4, u_index)

naive_W2_results = extract_results(5, u_index)
windowing_W2_results = extract_results(6, u_index)
smoothing_W2_results = extract_results(7, u_index)
concentration_W2_results = extract_results(8, u_index)

intersection_W2_results = extract_results(9, u_index)

function display_extracted_results(name, extracted_results)
    μ = mean(skipmissing(extracted_results[1]))
    σ = sem(skipmissing(extracted_results[1]))

    println(name*": $μ ± $σ")

    display(sort(collect(pairs(countmap(eachrow(hcat(extracted_results[2], extracted_results[3]))))), by = x->x.second, rev = true))

    #display(count(ismissing, extracted_results[1]))

end

display_extracted_results("Naive W1", naive_W1_results)
display_extracted_results("Windowing W1", windowing_W1_results)
display_extracted_results("Smoothing W1", smoothing_W1_results)
display_extracted_results("Concentration W1", concentration_W1_results)

display_extracted_results("Naive W2", naive_W2_results)
display_extracted_results("Windowing W2", windowing_W2_results)
display_extracted_results("Smoothing W2", smoothing_W2_results)
display_extracted_results("Concentration W2", concentration_W2_results)

display_extracted_results("Intersection W2", intersection_W2_results)