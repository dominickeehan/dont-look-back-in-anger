using CSV

repetitions = 10

W₂_costs = []
for job_number in 1:10
    results_file = CSV.File("$job_number.csv", header = false, skipto = 1)
    W₂_costs = [W₂_costs; [row.Column1 for row in Iterators.take(results_file, repetitions)]]
end

smoothing_costs = []
for job_number in 1:10
    results_file = CSV.File("$job_number.csv", header = false, skipto = 1+1*repetitions)
    smoothing_costs = [smoothing_costs; [row.Column1 for row in Iterators.take(results_file, repetitions)]]
end




