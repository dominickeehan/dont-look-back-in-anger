using CSV, Statistics, StatsBase

repetitions = 1
number_of_jobs = 10000

using ProgressBars

function extract_results(skipto)
    costs = zeros(number_of_jobs)
    parameter_1s = zeros(number_of_jobs)
    parameter_2s = zeros(number_of_jobs)
    #times = []
    
    Threads.@threads for job_number in ProgressBar(0:number_of_jobs-1)
        results_file = CSV.File("results/$job_number.csv", header = false, skipto = skipto*repetitions)
        costs[job_number+1], parameter_1s[job_number+1], parameter_2s[job_number+1] = 
            [[row.Column1, row.Column2, row.Column3] for row in Iterators.take(results_file, repetitions)][1]

    end

    return costs, parameter_1s, parameter_2s #, times
end

naive_SP_results = extract_results(1)
windowing_SP_results = extract_results(2)
smoothing_SP_results = extract_results(3)

naive_DRO_results = extract_results(4)
windowing_DRO_results = extract_results(5)
smoothing_DRO_results = extract_results(6)
concentration_DRO_results = extract_results(7)

intersection_DRO_results = extract_results(8)



function display_extracted_results(name, extracted_results)
    mean_ = mean(extracted_results[1])
    sem_ = sem(extracted_results[1])

    println(name*": $mean_ ± $sem_")

    #display(sort(collect(pairs(countmap(eachrow(hcat(extracted_results[2], extracted_results[3]))))), by = x->x.second, rev = true))

end

display_extracted_results("Naive SP", naive_SP_results)
display_extracted_results("Windowing SP", windowing_SP_results)
display_extracted_results("Smoothing SP", smoothing_SP_results)

display_extracted_results("Naive DRO", naive_DRO_results)
display_extracted_results("Windowing DRO", windowing_DRO_results)
display_extracted_results("Smoothing DRO", smoothing_DRO_results)
display_extracted_results("Concentration DRO", concentration_DRO_results)

display_extracted_results("Intersection DRO", intersection_DRO_results)