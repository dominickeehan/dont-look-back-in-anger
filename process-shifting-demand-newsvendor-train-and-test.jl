using CSV

repetitions = 100

concentration_costs = []
for job_number in 0:99
    results_file = CSV.File("results/$job_number.csv", header = false, skipto = 1)
    concentration_costs = [concentration_costs; [row.Column1 for row in Iterators.take(results_file, repetitions)]]
end

intersection_costs = []
for job_number in 0:99
    results_file = CSV.File("results/$job_number.csv", header = false, skipto = 1+4*repetitions)
    intersection_costs = [intersection_costs; [row.Column1 for row in Iterators.take(results_file, repetitions)]]
end

smoothing_costs = []
for job_number in 0:99
    results_file = CSV.File("results/$job_number.csv", header = false, skipto = 1+1*repetitions)
    smoothing_costs = [smoothing_costs; [row.Column1 for row in Iterators.take(results_file, repetitions)]]
end

windowing_costs = []
for job_number in 0:99
    results_file = CSV.File("results/$job_number.csv", header = false, skipto = 1+2*repetitions)
    windowing_costs = [windowing_costs; [row.Column1 for row in Iterators.take(results_file, repetitions)]]
end

naive_costs = []
for job_number in 0:99
    results_file = CSV.File("results/$job_number.csv", header = false, skipto = 1+3*repetitions)
    naive_costs = [naive_costs; [row.Column1 for row in Iterators.take(results_file, repetitions)]]
end

naive_mean = mean(naive_costs)
naive_sem = sem(naive_costs)
println("Naive W₂DRO: $naive_mean ± $naive_sem")

windowing_mean = mean(windowing_costs)
windowing_sem = sem(windowing_costs)
println("Windowing W₂DRO: $windowing_mean ± $windowing_sem")

smoothing_mean = mean(smoothing_costs)
smoothing_sem = sem(smoothing_costs)
println("Smoothing W₂DRO: $smoothing_mean ± $smoothing_sem")

concentration_mean = mean(concentration_costs)
concentration_sem = sem(concentration_costs)
println("Concentration W₂DRO: $concentration_mean ± $concentration_sem")

intersection_mean = mean(intersection_costs)
intersection_sem = sem(intersection_costs)
println("Intersection W₂DRO: $intersection_mean ± $intersection_sem")