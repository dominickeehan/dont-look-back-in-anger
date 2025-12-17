# "C:\Program Files\7-Zip\7z.exe" e "C:\Users\dkee331\Documents\repositories\dont-look-back-in-anger\newsvendor-data\dominic(10).zip" -r -o"C:\Users\dkee331\Documents\repositories\dont-look-back-in-anger\newsvendor-data" *.csv
# "C:\Program Files\7-Zip\7z.exe" e "C:\Users\domin\Documents\repositories\dont-look-back-in-anger\newsvendor-data\dom-small.zip" -r -o"C:\Users\domin\Documents\repositories\dont-look-back-in-anger\newsvendor-data" *.csv

using CSV, Statistics, StatsBase 
using ProgressBars
using Plots, Measures


number_of_jobs_per_drift = 1000 # (Code assumes number_of_repetitions = 1.)

number_of_consumers = 1000.0
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
                local job_index = 999*(drift_index-1)+job
                local results_file = CSV.File("newsvendor-data/$job_index.csv", header=false, skipto=skipto)

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

        gr(size = (275+6+8,183+6+9).*sqrt(3))

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
                topmargin = 9pt,
                leftmargin = 6pt,
                bottommargin = 6pt,
                rightmargin = 0pt,
                #legend = :bottom,
                )

        fillalpha = 0.1

        #normaliser, normaliser_sems = extract_line_to_plot(2)
        normaliser, normaliser_sems = ([183.0128717113633, 184.62990450375528, 183.40472107860836, 185.59454010728734, 183.76827749418942, 187.32362212419574, 184.4278583379568, 188.0999075519978, 193.4904629201685, 222.57161468174186, 286.20874812273564], [1.5431727828948911, 1.6089016508440883, 1.5099443075243746, 1.6342467722998617, 1.5343725877619685, 1.9494327403324878, 2.505026027233826, 3.577923754616864, 4.316775872353911, 3.680758949538472, 5.096065509362292])

        #expected_costs, sems = extract_line_to_plot(1)
        expected_costs, sems = ([168.83063744707127, 169.34881296541704, 170.92225825904754, 174.3865026857558, 179.3032432189207, 188.60696013719405, 203.4492693328096, 233.16117125747363, 317.0979041144834, 405.59089367452935, 458.4503858757774], [0.18779161870807576, 0.22705589735696696, 0.2904076644205518, 0.4751913814504601, 0.8408575077679022, 1.3931291464996947, 2.451467266026189, 4.112242927539038, 7.7954865798475925, 9.657921312331677, 9.61049489232579])
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

        #expected_costs, sems = extract_line_to_plot(3)
        expected_costs, sems = ([216.80729917513506, 216.87654667489082, 217.0624431871434, 217.4212817033139, 213.1293153912676, 212.17487744997706, 201.5934460319634, 196.96920061693413, 206.94290539064286, 238.48176782883837, 289.5647069821513], [1.2505500203551956, 1.308434281147745, 1.4402673189889976, 1.5118038836246341, 1.5757866722920841, 2.078265580438921, 2.9069231138321876, 3.912518648580798, 4.856413233993833, 5.142819081674797, 5.923156544727841])
        plot!(drifts, expected_costs./normaliser, ribbon = sems./normaliser, fillalpha = fillalpha,
                color = palette(:tab10)[1],
                linestyle = :solid,
                markershape = :circle,
                markersize = 4.0,
                markerstrokewidth = 0.0,
                label = "Intersection")

        #expected_costs, sems = extract_line_to_plot(4)
        expected_costs, sems = ([187.51233637236757, 190.90639326972726, 186.70373288890664, 188.12271562022147, 185.20360406562693, 189.2095586603273, 186.44553631884438, 185.14424439261634, 191.51037970814923, 212.25399700394672, 270.5625413109855], [1.777211013495511, 1.9623800600082664, 1.7824646845050895, 1.786918218227337, 1.7170315339178344, 2.14547592262683, 2.8228010892944435, 3.784045044941464, 4.770977841375317, 3.961253568307639, 6.161035159464617])
        plot!(drifts, expected_costs./normaliser, ribbon = sems./normaliser, fillalpha = fillalpha,
                color = palette(:tab10)[2],
                linestyle = :dash,
                markershape = :diamond,
                markersize = 4.0,
                markerstrokewidth = 0.0,
                label = "Weighted")

        xticks!([1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e0])
        ylims!((0.9, 1.2))
        xlims!((0.99999*drifts[5], 1.00001*drifts[end]))

        display(plt)

        #savefig(plt, "figures/average-train-and-test-next-period-expected-cost-17-12.pdf")