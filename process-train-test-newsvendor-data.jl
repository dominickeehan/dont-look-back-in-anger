# "C:\Program Files\7-Zip\7z.exe" e "C:\Users\dkee331\Documents\repositories\dont-look-back-in-anger\newsvendor-data\dominic(10).zip" -r -o"C:\Users\dkee331\Documents\repositories\dont-look-back-in-anger\newsvendor-data" *.csv

using CSV, Statistics, StatsBase 
using ProgressBars
using Plots, Measures


number_of_jobs_per_u = 1000 # (Code assumes [number of] repetitions = 1.)

U = [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2]#, 5e-2]

history_length = 100
ε = [0; LinRange(1e0,1e1,10); LinRange(2e1,1e2,9); LinRange(2e2,1e3,9); LinRange(2e3,1e4,9); LinRange(2e4,1e5,9)]
s = [round.(Int, LinRange(1,10,10)); round.(Int, LinRange(12,30,10)); round.(Int, LinRange(33,60,10)); round.(Int, LinRange(64,100,10))]
LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))
α = [0; LogRange(1e-4,1e0,39)]
ϱ╱ε = [0; LogRange(1e-4,1e0,39)]

ambiguity_radii = [[0], [0], [0], ε, ε, ε, ε, ε, ε, ε, ε, ε[2:end]]
weight_parameters = [[history_length], s, α, [history_length], s, α, ϱ╱ε, [history_length], s, α, ϱ╱ε, ϱ╱ε,]

function extract_ex_post_expected_cost(method_index, u_index)

        length_ambiguity_radii = length(ambiguity_radii[method_index])
        length_weight_parameters = length(weight_parameters[method_index])

        costs = [zeros((length_ambiguity_radii,length_weight_parameters)) for _ in 1:number_of_jobs_per_u]
        is_indice_missing = falses(number_of_jobs_per_u)

        skipto = sum(length(ambiguity_radii[i])*length(weight_parameters[i]) for i in 1:method_index-1; init=0)+1+1 # (Second +1 to ignore header)
        take = length_ambiguity_radii*length_weight_parameters       

        Threads.@threads for job in 0:number_of_jobs_per_u-1
                local job_index = 999*(u_index-1)+job
                local results_file = CSV.File("newsvendor-data/$job_index.csv", header=false, skipto=skipto)

                try
                        local data = [row.Column9 for row in Iterators.take(results_file, take)]
                        costs[job+1] = reshape(data, length_weight_parameters, length_ambiguity_radii)'

                catch
                        is_indice_missing[job+1] = true

                end
        end

        costs = costs[.!is_indice_missing]

        ambiguity_radius_index, weight_parameter_index = Tuple(argmin(mean(costs)))
        minimal_costs = [costs[i][ambiguity_radius_index, weight_parameter_index] for i in eachindex(costs)]

        return mean(minimal_costs), sem(minimal_costs)
end


function extract_train_test_objective_values_and_expected_costs(method_index, u_index)

        length_ambiguity_radii = length(ambiguity_radii[method_index])
        length_weight_parameters = length(weight_parameters[method_index])

        training_average_costs = [zeros((length_ambiguity_radii,length_weight_parameters)) for _ in 1:number_of_jobs_per_u]
        objective_values = [zeros((length_ambiguity_radii,length_weight_parameters)) for _ in 1:number_of_jobs_per_u]
        test_expected_costs = [zeros((length_ambiguity_radii,length_weight_parameters)) for _ in 1:number_of_jobs_per_u]
        is_indice_missing = falses(number_of_jobs_per_u)

        skipto = sum(length(ambiguity_radii[i])*length(weight_parameters[i]) for i in 1:method_index-1; init=0)+1+1 # (Second +1 to ignore header)
        take = length_ambiguity_radii*length_weight_parameters       

        Threads.@threads for job in 0:number_of_jobs_per_u-1
                local job_index = 999*(u_index-1)+job
                local results_file = CSV.File("newsvendor-data/$job_index.csv", header=false, skipto=skipto)

                try
                        local training_average_cost_data, doubling_count_data, objective_values_data, test_expected_cost_data = eachcol(stack([[row.Column6, row.Column7, row.Column8, row.Column9] for row in Iterators.take(results_file, take)])')
                        training_average_cost_data[doubling_count_data .> 0] .= Inf
                        training_average_costs[job+1] = reshape(training_average_cost_data, length_ambiguity_radii, length_weight_parameters)
                        objective_values[job+1] = reshape(objective_values_data, length_ambiguity_radii, length_weight_parameters)

                        test_expected_costs[job+1] = reshape(test_expected_cost_data, length_ambiguity_radii, length_weight_parameters)

                catch
                        is_indice_missing[job+1] = true

                end
        end

        training_average_costs = training_average_costs[.!is_indice_missing]
        objective_values = objective_values[.!is_indice_missing]
        test_expected_costs = test_expected_costs[.!is_indice_missing]

        realised_objective_values = zeros(length(test_expected_costs))
        realised_costs = zeros(length(test_expected_costs))
        for i in eachindex(test_expected_costs)
                ambiguity_radius_index, weight_parameter_index = Tuple(argmin(training_average_costs[i]))
                realised_objective_values[i] = objective_values[i][ambiguity_radius_index, weight_parameter_index]
                realised_costs[i] = test_expected_costs[i][ambiguity_radius_index, weight_parameter_index]
        
        end

        #display(is_indice_missing)
        #display(realised_costs)

        return realised_objective_values, realised_costs
end


8787

#display(extract_ex_post_expected_cost(7, 6))

#display(extract_train_test_expected_cost(11, 6))
#display(extract_train_test_expected_cost(12, 6))

#throw = throw



#extract_expected_cost = extract_ex_post_expected_cost
#extract_expected_cost = extract_train_test_expected_cost

function extract_line_to_plot(method_index)

        expected_costs = zeros(length(U))
        sems = zeros(length(U))

        for u_index in ProgressBar(eachindex(U))
                #expected_costs[u_index], sems[u_index] = extract_ex_post_expected_cost(method_index, u_index)
                _, costs = extract_train_test_objective_values_and_expected_costs(method_index, u_index)
                expected_costs[u_index] = mean(costs)
                sems[u_index] = sem(costs)
                #expected_costs[u_index], sems[u_index] = extract_train_test_expected_cost(method_index, u_index)

        end

    return expected_costs, sems

end



if true # Plot some

        default() # Reset plot defaults.

        gr(size = (275+6,183+6).*sqrt(3))

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
                xlabel = "Distribution shift parameter, \$ρ′\$", 
                ylabel = "Expected cost",
                #ylabel = "Ex-post-optimal expected cost",
                topmargin = 0pt,
                leftmargin = 6pt,
                bottommargin = 6pt,
                rightmargin = 0pt,
                )

        fillalpha = 0.1

        #normalizer, normalizer_sems = extract_line_to_plot(3)
        normalizer, normalizer_sems = ([44.26806163356526, 44.64146019375224, 46.13589474158781, 47.525224406621035, 49.08395635413206, 59.58361944242603, 78.41465537013588, 100.69684593517886, 123.44445692049854, 286.5123360179635], [0.17301248629485963, 0.14035162893571035, 0.20718777959226034, 0.25339042491472796, 0.3188825171200088, 0.6107547117132267, 0.9607704724500018, 1.1541477711047032, 1.1628728450528392, 2.5615479953270617])

        #expected_costs, sems = extract_line_to_plot(1)
        expected_costs, sems = ([42.87706459429235, 44.15192191800951, 48.696929122988024, 54.87609383271609, 62.12184696064479, 118.80083147295987, 221.84701431183333, 324.42588950149303, 421.5265380821383, 910.5999275648368], [0.035739709058672756, 0.0871007407128359, 0.26949057371472923, 0.4907668747550239, 0.7592103227692515, 2.556446110715032, 5.275165716325518, 8.05787966599145, 10.250560159689947, 27.70619993027752])
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[1],
                linestyle = :solid,
                markershape = :circle,
                markersize = 4,
                markerstrokewidth = 0,
                label = "Unweighted (\$ε=0\$)")#"Naïve \$(ε=0)\$")

        expected_costs, sems = normalizer, normalizer_sems
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[2],
                linestyle = :dash,
                markershape = :diamond,
                markersize = 4,
                markerstrokewidth = 0,
                label = "Smoothing (\$ε=0\$)")

        #expected_costs, sems = extract_line_to_plot(12)
        expected_costs, sems = ([46.89051102710592, 46.80519746412745, 48.30955329059213, 49.236172128682085, 50.743627230150715, 58.98498313090304, 74.87442886743092, 95.12314854477603, 114.74273519275445, 251.2954419201019], [0.29108574867941117, 0.2835947321255175, 0.3206981891435156, 0.3526277354988551, 0.38707587432146157, 0.5935867108376222, 0.7922520951319628, 1.0227371406000576, 1.0886611582078976, 2.980690218127019])
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[7],
                linestyle = :dashdot,
                markershape = :pentagon,
                markersize = 4,
                markerstrokewidth = 0,
                label = "Intersections")

        #expected_costs, sems = extract_line_to_plot(11)
        expected_costs, sems = ([45.2325160911206, 45.54197932949779, 46.85240850735009, 48.14695798234909, 50.14248888792851, 59.42027868613782, 75.71901229647328, 91.18038137728956, 106.92100639840483, 220.3178730325334], [0.20454258559584673, 0.20452832287244888, 0.24755559634011573, 0.2780938338579865, 0.3854379322096642, 0.5606726981653295, 0.7534178905852787, 0.8071049341061514, 0.7781022250013478, 1.5216141465892843])
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[9],
                linestyle = :dot,
                markershape = :star4,
                markersize = 5,
                markerstrokewidth = 0,
                label = "Weighted")

        ylims!((0.7, 1.3))
        xlims!((U[1], U[end]))

        display(plt)

        #savefig(plt, "figures/train-test-expected-cost.pdf")

end


if true # Plot some

        default() # Reset plot defaults.

        gr(size = (275+6,183+6).*sqrt(3))

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
                xlabel = "Distribution shift parameter, \$ρ′\$", 
                ylabel = "Expected cost",
                #ylabel = "Ex-post-optimal expected cost",
                topmargin = 0pt,
                leftmargin = 6pt,
                bottommargin = 6pt,
                rightmargin = 0pt,
                )

        fillalpha = 0.1

        normalizer, normalizer_sems = extract_line_to_plot(5)

        expected_costs, sems = normalizer, normalizer_sems
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[1],
                linestyle = :solid,
                markershape = :circle,
                markersize = 4,
                markerstrokewidth = 0,
                label = "Windowing (\$W_1\$)")

        expected_costs, sems = extract_line_to_plot(6)
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[2],
                linestyle = :dash,
                markershape = :diamond,
                markersize = 4,
                markerstrokewidth = 0,
                label = "Smoothing (\$W_1\$)")

        expected_costs, sems = extract_line_to_plot(7)
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[3],
                linestyle = :dashdot,
                markershape = :pentagon,
                markersize = 4,
                markerstrokewidth = 0,
                label = "Weighted (\$W_1\$)")

        #ylims!((0.7, 1.3))
        #xlims!((U[1], U[end]))

        display(plt)

end



if true # Plot some

        default() # Reset plot defaults.

        gr(size = (275+6,183+6).*sqrt(3))

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
                xlabel = "Distribution shift parameter, \$ρ′\$", 
                ylabel = "Expected cost",
                #ylabel = "Ex-post-optimal expected cost",
                topmargin = 0pt,
                leftmargin = 6pt,
                bottommargin = 6pt,
                rightmargin = 0pt,
                )

        fillalpha = 0.1

        normalizer, normalizer_sems = extract_line_to_plot(9)

        expected_costs, sems = normalizer, normalizer_sems
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[1],
                linestyle = :solid,
                markershape = :circle,
                markersize = 4,
                markerstrokewidth = 0,
                label = "Windowing (\$W_2\$)")

        expected_costs, sems = extract_line_to_plot(10)
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[2],
                linestyle = :dash,
                markershape = :diamond,
                markersize = 4,
                markerstrokewidth = 0,
                label = "Smoothing (\$W_2\$)")

        expected_costs, sems = extract_line_to_plot(11)
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[3],
                linestyle = :dashdot,
                markershape = :pentagon,
                markersize = 4,
                markerstrokewidth = 0,
                label = "Weighted (\$W_2\$)")

        #ylims!((0.7, 1.3))
        #xlims!((U[1], U[end]))

        display(plt)

end






























if false # Plot all

        default() # Reset plot defaults.

        gr(size = (600,400))

        font_family = "Computer Modern"
        primary_font = Plots.font(font_family, pointsize = 17)
        secondary_font = Plots.font(font_family, pointsize = 11)
        legend_font = Plots.font(font_family, pointsize = 13)

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
                fontfamily = font_family,
                guidefont = primary_font,
                tickfont = secondary_font,
                legendfont = legend_font)

        plt = plot(xscale = :log10, #yscale = :log10,
                xlabel = "Shift, \$u\$", 
                ylabel = "Expected cost (normalized)",)

        fillalpha = 0.1

        normalizer, normalizer_sems = extract_line_to_plot(3)

        expected_costs, sems = extract_line_to_plot(1)
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[1],
                linestyle = :solid,
                label = nothing)

        expected_costs, sems = extract_line_to_plot(2)
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[2],
                linestyle = :solid,
                label = nothing)
        expected_costs, sems = normalizer, normalizer_sems
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[3],
                linestyle = :solid,
                label = nothing)

        expected_costs, sems = extract_line_to_plot(8)
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[1],
                linestyle = :dash,
                label = nothing)

        expected_costs, sems = extract_line_to_plot(9)
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[2],
                linestyle = :dash,
                label = nothing)
        expected_costs, sems = extract_line_to_plot(10)
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[3],
                linestyle = :dash,
                label = nothing)
        expected_costs, sems = extract_line_to_plot(11)
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[4],
                linestyle = :dash,
                label = nothing)

        expected_costs, sems = extract_line_to_plot(12)
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[5],
                linestyle = :dash,
                label = nothing)

        ylims!((0.75, 1.5))
        xlims!((U[1], U[end]))

        #plot!(legend_columns = 2)
        #plot!([0,-1], [-1,-1], linestyle = :solid, color = :white, label = " ")

        plot!([0,-1], [-1,-1], linestyle = :solid, color = :black, label = "SO")
        plot!([0,-1], [-1,-1], linestyle = :dash, color = :black, label = "\$W_2\$")

        scatter!([0,-1], [-1,-1], markershape = :circle, markerstrokecolor = palette(:tab10)[1], color = palette(:tab10)[1], label = "Naïve")
        scatter!([0,-1], [-1,-1], markershape = :circle, markerstrokecolor = palette(:tab10)[2], color = palette(:tab10)[2], label = "Windowing")
        scatter!([0,-1], [-1,-1], markershape = :circle, markerstrokecolor = palette(:tab10)[3], color = palette(:tab10)[3], label = "Smoothing")
        scatter!([0,-1], [-1,-1], markershape = :circle, markerstrokecolor = palette(:tab10)[4], color = palette(:tab10)[4], label = "Concentration")
        scatter!([0,-1], [-1,-1], markershape = :circle, markerstrokecolor = palette(:tab10)[5], color = palette(:tab10)[5], label = "Intersection")

        plot!(legend = :topleft)

        display(plt)

        #savefig(plt, "figures/to-discuss-1.pdf")

end


if false

        function extract_objective_value_percentage_disappointments(method_index, u_index)

                objective_values, expected_costs = extract_train_test_objective_values_and_expected_costs(method_index, u_index)

                display(min(objective_values...))

                return 100*(expected_costs - objective_values)./(objective_values)

        end

        u_index = 6
        objective_value_disappointments = extract_objective_value_percentage_disappointments(12, u_index)
        #objective_value_disappointments = objective_value_disappointments[objective_value_disappointments .>= 0]


        #bin_locations = [1e1,2e1,3e1,4e1,5e1,6e1,7e1,8e1,9e1,1e2,2e2,3e2,4e2,5e2,6e2,7e2,8e2,9e2,1e3,2e3,3e3,4e3,5e3,6e3,7e3,8e3,9e3]

        for objective_value_disappointment_index in eachindex(objective_value_disappointments)
                if objective_value_disappointments[objective_value_disappointment_index] <= 1
                        #objective_value_disappointments[objective_value_disappointment_index] = 1.1
                end
        end

        default() # Reset plot defaults.

        gr(size = (600,400))

        font_family = "Computer Modern"
        primary_font = Plots.font(font_family, pointsize = 17)
        secondary_font = Plots.font(font_family, pointsize = 11)
        legend_font = Plots.font(font_family, pointsize = 13)

        default(framestyle = :box,
                grid = true,
                #gridlinewidth = 1.0,
                gridalpha = 0.075,
                #minorgrid = true,
                #minorgridlinewidth = 1.0, 
                #minorgridalpha = 0.075,
                #minorgridlinestyle = :dash,
                tick_direction = :in,
                #xminorticks = 0, 
                #yminorticks = 0,
                fontfamily = font_family,
                guidefont = primary_font,
                tickfont = secondary_font,
                legendfont = legend_font)

        plt = plot(
                xlims=(-100,300),
                #xscale = :log10,
                #xaxis=(:log10, [1, :auto]),
                ylabel="Frequency (normalized)",
                xlabel="Out-of-sample disappointment (%)",
                #xticks = ([1,10,100,1000,10000])
                )

        bins = LogRange(1,10000,10)
        fillalpha = 0.1



        stephist!(objective_value_disappointments,
                #bins = bins,
                bins = 300,
                color = palette(:tab10)[7],
                linestyle = :dashdot,
                normalize = :pdf,
                fill = true,
                fillalpha = fillalpha,
                #xscale=:log10, 
                label = "Intersections")

        vspan!([mean(objective_value_disappointments)-sem(objective_value_disappointments), mean(objective_value_disappointments)+sem(objective_value_disappointments)], 
                color = palette(:tab10)[7],
                alpha = fillalpha,
                label = nothing)
        vline!([mean(objective_value_disappointments)],
                color = palette(:tab10)[7],
                linestyle = :dashdot,
                label = nothing)


        objective_value_disappointments = extract_objective_value_percentage_disappointments(11, u_index)
        #objective_value_disappointments = objective_value_disappointments[objective_value_disappointments .>= 0]

        for objective_value_disappointment_index in eachindex(objective_value_disappointments)
                if objective_value_disappointments[objective_value_disappointment_index] <= 1
                        #objective_value_disappointments[objective_value_disappointment_index] = 1.1
                end
        end

        stephist!(objective_value_disappointments,
                #bins = bins,
                bins = 100,
                color = palette(:tab10)[9],
                linestyle = :dot,
                normalize = :pdf,
                fill = true,
                #xscale=:log10, 
                fillalpha = fillalpha, 
                label = "Concentration")

        vspan!([mean(objective_value_disappointments)-sem(objective_value_disappointments), mean(objective_value_disappointments)+sem(objective_value_disappointments)], 
                color = palette(:tab10)[9],
                alpha = fillalpha,
                label = nothing)
        vline!([mean(objective_value_disappointments)],
                color = palette(:tab10)[9],
                linestyle = :dot,
                label = nothing)


        #stephist!(skipmissing(extract_histogram_to_plot(12, u_index)), bins=bins, color=palette(:tab10)[5], linestyle=:dash, normalize=:pdf, fill=true, fillalpha=0.1, label="\$W_2\$ Intersections")
        #vspan!([-sem(skipmissing(extract_histogram_to_plot(12, u_index)))+mean(skipmissing(extract_histogram_to_plot(12, u_index))),sem(skipmissing(extract_histogram_to_plot(12, u_index)))+mean(skipmissing(extract_histogram_to_plot(12, u_index)))], color=palette(:tab10)[5], alpha=0.1, label=nothing)
        #vline!([mean(skipmissing(extract_histogram_to_plot(12, u_index)))], color=palette(:tab10)[5], linestyle=:dash, label=nothing)
        display(plt)

        #x = extract_histogram_to_plot(12, u_index)
        #min(x...)

        #ints = identity.(skipmissing(extract_histogram_to_plot(11, u_index)))
        #display(mean(ints[ints .>= 0]))

        #ints = identity.(skipmissing(extract_histogram_to_plot(12, u_index)))
        #display(mean(ints[ints .>= 0]))

        #savefig(plt, "figures/to-discuss-3.pdf")
end