# "C:\Program Files\7-Zip\7z.exe" e "C:\driftssers\dkee331\Documents\repositories\dont-look-back-in-anger\newsvendor-data\dominic(10).zip" -r -o"C:\driftssers\dkee331\Documents\repositories\dont-look-back-in-anger\newsvendor-data" *.csv

using CSV, Statistics, StatsBase 
using ProgressBars
using Plots, Measures


number_of_jobs_per_drift = 1000 # (Code assumes number_of_repetitions = 1.)

drifts = [1.00e-3, 1.79e-3, 3.16e-3, 5.62e-3, 1.00e-2, 1.79e-2, 3.16e-2, 5.62e-2, 1.00e-1, 1.79e-1, 3.16e-1] # exp10.(LinRange(log10(1),log10(10),5))

history_length = 100

LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))
ε = number_of_consumers*unique([0.0; LinRange(1.0e-3,1.0e-2,10); LinRange(1.0e-2,1.0e-1,10); LinRange(1.0e-1,1.0e-0,10)])
s = unique(round.(Int, LogRange(1,history_length,30)))
α = [0.0; LogRange(1.0e-4,1.0e0,30)]
ρ╱ε = [0.0; LogRange(1.0e-4,1.0e0,30)]
intersection_ε = number_of_consumers*unique([LinRange(1.0e-3,1.0e-2,10); LinRange(1.0e-2,1.0e-1,10); LinRange(1.0e-1,1.0e-0,10)])
intersection_ρ╱ε = [0.0; LogRange(1.0e-4,1.0e0,30)]

ambiguity_radii = [[0], [0], intersection_ε, ε]
weight_parameters = [[history_length], α, intersection_ρ╱ε, ρ╱ε]

function extract_train_test_objective_values_and_expected_next_period_costs(method_index, drift_index)

        length_ambiguity_radii = length(ambiguity_radii[method_index])
        length_weight_parameters = length(weight_parameters[method_index])

        train_average_costs = [zeros((length_ambiguity_radii,length_weight_parameters)) for _ in 1:number_of_jobs_per_drift]
        objective_values = [zeros((length_ambiguity_radii,length_weight_parameters)) for _ in 1:number_of_jobs_per_drift]
        test_expected_costs = [zeros((length_ambiguity_radii,length_weight_parameters)) for _ in 1:number_of_jobs_per_drift]
        #are_indices_missing = falses(number_of_jobs_per_drift)

        skipto = sum(length(ambiguity_radii[i])*length(weight_parameters[i]) for i in 1:method_index-1; init=0)+1+1 # (Second +1 to ignore header)
        take = length_ambiguity_radii*length_weight_parameters       

        Threads.@threads for job in 0:number_of_jobs_per_drift-1
                local job_index = (number_of_jobs_per_drift-1)*(drift_index-1)+job
                local results_file = CSV.File("drifting-newsvendor-data/27-09-25/$job_index.csv", header=false, skipto=skipto)

                #try
                        local train_average_cost_data, doubling_count_data, objective_values_data, test_expected_cost_data = eachcol(stack([[row.Column6, row.Column7, row.Column8, row.Column9] for row in Iterators.take(results_file, take)])')
                        
                        # Exclude solver issue runs for intersection.
                        # train_average_cost_data[doubling_count_data .> 0] .= Inf
                        #replace!(train_average_cost_data, NaN => Inf)
                        #replace!(test_expected_cost_data, NaN => Inf)
                        #train_average_cost_data[test_expected_cost_data .> 1000] .= Inf
                        
                        train_average_costs[job+1] = reshape(train_average_cost_data, length_ambiguity_radii, length_weight_parameters)
                        objective_values[job+1] = reshape(objective_values_data, length_ambiguity_radii, length_weight_parameters)

                        test_expected_costs[job+1] = reshape(test_expected_cost_data, length_ambiguity_radii, length_weight_parameters)

                #catch
                #        are_indices_missing[job+1] = true

                #end
        end

        #train_average_costs = train_average_costs[.!are_indices_missing]
        #objective_values = objective_values[.!are_indices_missing]
        #test_expected_costs = test_expected_costs[.!are_indices_missing]

        realised_objective_values = zeros(length(test_expected_costs))
        realised_next_period_expected_costs = zeros(length(test_expected_costs))
        for i in eachindex(test_expected_costs)
                ambiguity_radius_index, weight_parameter_index = Tuple(argmin(train_average_costs[i]))
                realised_objective_values[i] = objective_values[i][ambiguity_radius_index, weight_parameter_index]
                realised_next_period_expected_costs[i] = test_expected_costs[i][ambiguity_radius_index, weight_parameter_index]
        
        end

        #display(are_indices_missing)
        #display(realised_next_period_expected_costs)

        return realised_objective_values, realised_next_period_expected_costs
end


8787

#display(extract_ex_post_expected_cost(7, 6))

#display(extract_train_test_expected_cost(11, 6))
#println(max.(extract_train_test_objective_values_and_expected_costs(12, 10)[2]...))

#throw = throw



#extract_expected_cost = extract_ex_post_expected_cost
#extract_expected_cost = extract_train_test_expected_cost

function extract_line_to_plot(method_index)

        expected_next_period_expected_costs = zeros(length(drifts))
        sems = zeros(length(drifts))

        for drift_index in ProgressBar(eachindex(drifts))
            _, next_period_expected_costs = extract_train_test_objective_values_and_expected_next_period_costs(method_index, drift_index)
            expected_next_period_expected_costs[drift_index] = mean(next_period_expected_costs)
            sems[drift_index] = sem(next_period_expected_costs)

        end

    return expected_next_period_expected_costs, sems

end



#if true # Plot some

        default() # Reset plot defaults.

        gr(size = (275+6+8,183+6).*sqrt(3))

        fontfamily = "Computer Modern"

        default(framestyle = :box,
                grid = true,
                #gridlinewidth = 1.0,
                gridalpha = 0.075,
                minorgrid = true,
                #minorgridlinewidth = 1.0, 
                minorgridalpha = 0.075,
                minorgridlinestyle = :dash,
                tick_direction = :in,
                xminorticks = 9, 
                yminorticks = 0,
                fontfamily = fontfamily,
                guidefont = Plots.font(fontfamily, pointsize = 12),
                legendfont = Plots.font(fontfamily, pointsize = 11),
                tickfont = Plots.font(fontfamily, pointsize = 10))

        plt = plot(xscale = :log10, #yscale = :log10,
                xlabel ="Binomial drift parameter, \$δ\$", 
                ylabel = "Average train-and-test next-period\nexpected cost (relative to smoothing)",
                topmargin = 0pt,
                leftmargin = 6pt,
                bottommargin = 6pt,
                rightmargin = 0pt,
                )

        fillalpha = 0.1

        normaliser, normaliser_sems = extract_line_to_plot(2)
        #normaliser, normaliser_sems = ([44.26806163356526, 44.64146019375224, 46.13589474158781, 47.525224406621035, 49.08395635413206, 59.58361944242603, 78.41465537013588, 100.69684593517886, 123.44445692049854, 286.5123360179635], [0.17301248629485963, 0.14035162893571035, 0.20718777959226034, 0.25339042491472796, 0.3188825171200088, 0.6107547117132267, 0.9607704724500018, 1.1541477711047032, 1.1628728450528392, 2.5615479953270617])

        expected_costs, sems = extract_line_to_plot(1)
        #expected_costs, sems = ([42.87706459429235, 44.15192191800951, 48.696929122988024, 54.87609383271609, 62.12184696064479, 118.80083147295987, 221.84701431183333, 324.42588950149303, 421.5265380821383, 910.5999275648368], [0.035739709058672756, 0.0871007407128359, 0.26949057371472923, 0.4907668747550239, 0.7592103227692515, 2.556446110715032, 5.275165716325518, 8.05787966599145, 10.250560159689947, 27.70619993027752])
        plot!(drifts, expected_costs./normaliser, ribbon = sems./normaliser, fillalpha = fillalpha,
                color = palette(:tab10)[7],
                linestyle = :dashdot,
                markershape = :pentagon,
                markersize = 4.0,
                markerstrokewidth = 0.0,
                label = "SAA (\$ε=0\$)")

        expected_costs, sems = normaliser, normaliser_sems
        plot!(drifts, expected_costs./normaliser, ribbon = sems./normaliser, fillalpha = fillalpha,
                color = palette(:tab10)[9],
                linestyle = :dot,
                linewidth = 1.2,
                markershape = :star4,
                markersize = 6.0,
                markerstrokewidth = 0.0,
                label = "Smoothing (\$ε=0\$)")

        expected_costs, sems = extract_line_to_plot(12)
        #expected_costs, sems = ([46.89051102710592, 46.80519746412745, 48.30955329059213, 49.236172128682085, 50.743627230150715, 58.98498313090304, 74.87442886743092, 95.12314854477603, 114.74273519275445, 251.2954419201019], [0.29108574867941117, 0.2835947321255175, 0.3206981891435156, 0.3526277354988551, 0.38707587432146157, 0.5935867108376222, 0.7922520951319628, 1.0227371406000576, 1.0886611582078976, 2.980690218127019])
        plot!(drifts, expected_costs./normaliser, ribbon = sems./normaliser, fillalpha = fillalpha,
                color = palette(:tab10)[1],
                linestyle = :solid,
                markershape = :circle,
                markersize = 4.0,
                markerstrokewidth = 0.0,
                label = "Intersection")

        expected_costs, sems = extract_line_to_plot(11)
        #expected_costs, sems = ([45.2325160911206, 45.54197932949779, 46.85240850735009, 48.14695798234909, 50.14248888792851, 59.42027868613782, 75.71901229647328, 91.18038137728956, 106.92100639840483, 220.3178730325334], [0.20454258559584673, 0.20452832287244888, 0.24755559634011573, 0.2780938338579865, 0.3854379322096642, 0.5606726981653295, 0.7534178905852787, 0.8071049341061514, 0.7781022250013478, 1.5216141465892843])
        plot!(drifts, expected_costs./normaliser, ribbon = sems./normaliser, fillalpha = fillalpha,
                color = palette(:tab10)[2],
                linestyle = :dash,
                markershape = :diamond,
                markersize = 4.0,
                markerstrokewidth = 0.0,
                label = "Weighted")

        xticks!([1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e0])
        ylims!((0.7, 1.3))
        xlims!((0.99999*drifts[1], 1.00001*drifts[end]))

        display(plt)

        savefig(plt, "figures/average-train-and-test-next-period-expected-cost.pdf")