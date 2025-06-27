# "C:\Program Files\7-Zip\7z.exe" e "C:\Users\dkee331\Documents\repositories\dont-look-back-in-anger\newsvendor-data\dominic(10).zip" -r -o"C:\Users\dkee331\Documents\repositories\dont-look-back-in-anger\newsvendor-data" *.csv

using CSV, Statistics, StatsBase 
using ProgressBars
using Plots, Measures


number_of_jobs_per_u = 1000 # (Code assumes [number of] repetitions = 1.)

U = [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2, 5e-2]

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


function extract_train_test_expected_cost(method_index, u_index)

        length_ambiguity_radii = length(ambiguity_radii[method_index])
        length_weight_parameters = length(weight_parameters[method_index])

        training_costs = [zeros((length_ambiguity_radii,length_weight_parameters)) for _ in 1:number_of_jobs_per_u]
        test_costs = [zeros((length_ambiguity_radii,length_weight_parameters)) for _ in 1:number_of_jobs_per_u]
        is_indice_missing = falses(number_of_jobs_per_u)

        skipto = sum(length(ambiguity_radii[i])*length(weight_parameters[i]) for i in 1:method_index-1; init=0)+1+1 # (Second +1 to ignore header)
        take = length_ambiguity_radii*length_weight_parameters       

        Threads.@threads for job in 0:number_of_jobs_per_u-1
                local job_index = 999*(u_index-1)+job
                local results_file = CSV.File("newsvendor-data/$job_index.csv", header=false, skipto=skipto)

                try
                        local training_cost_data, doubling_count_data, test_cost_data = eachcol(stack([[row.Column6, row.Column7, row.Column9] for row in Iterators.take(results_file, take)])')
                        training_cost_data[doubling_count_data .> 0] .= Inf
                        training_costs[job+1] = reshape(training_cost_data, length_ambiguity_radii, length_weight_parameters)

                        test_costs[job+1] = reshape(test_cost_data, length_ambiguity_radii, length_weight_parameters)

                catch
                        is_indice_missing[job+1] = true

                end
        end

        training_costs = training_costs[.!is_indice_missing]
        test_costs = test_costs[.!is_indice_missing]

        realised_costs = zeros(length(test_costs))
        for i in eachindex(test_costs)
                ambiguity_radius_index, weight_parameter_index = Tuple(argmin(training_costs[i]))
                realised_costs[i] = test_costs[i][ambiguity_radius_index, weight_parameter_index]
        
        end

        #display(is_indice_missing)
        #display(realised_costs)

        return mean(realised_costs), sem(realised_costs)
end


8787

#display(extract_ex_post_expected_cost(12, 5))
#display(extract_train_test_expected_cost(12, 5))

#throw = throw



#extract_expected_cost = extract_ex_post_expected_cost
#extract_expected_cost = extract_train_test_expected_cost

function extract_line_to_plot(method_index)

        expected_costs = zeros(length(U))
        sems = zeros(length(U))

        for u_index in ProgressBar(eachindex(U))
                expected_costs[u_index], sems[u_index] = extract_ex_post_expected_cost(method_index, u_index)
                #expected_costs[u_index], sems[u_index] = extract_train_test_expected_cost(method_index, u_index)

        end

    return expected_costs, sems

end



if true # Plot some

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
                xlabel = "Distribution shift, \$u\$", 
                #ylabel = "Train/test expected cost",
                ylabel = "Ex-post-optimal expected cost",
                topmargin = 10pt)

        fillalpha = 0.1

        normalizer, normalizer_sems = extract_line_to_plot(3)

        expected_costs, sems = extract_line_to_plot(1)
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[8],
                linestyle = :dot,
                #markershape = :diamond,
                #markersize = 4,
                #markerstrokewidth = 0,
                label = "Naïve \$(ε=0)\$")

        expected_costs, sems = normalizer, normalizer_sems
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[1],
                linestyle = :solid,
                markershape = :circle,
                markersize = 4,
                markerstrokewidth = 0,
                label = "Smoothing \$(ε=0)\$")

        expected_costs, sems = extract_line_to_plot(12)
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[7],
                linestyle = :dash,
                markershape = :diamond,
                markersize = 4,
                markerstrokewidth = 0,
                label = "Intersections")

        expected_costs, sems = extract_line_to_plot(11)
        plot!(U, expected_costs./normalizer, ribbon = sems./normalizer, fillalpha = fillalpha,
                color = palette(:tab10)[9],
                linestyle = :dashdot,
                markershape = :pentagon,
                markersize = 4,
                markerstrokewidth = 0,
                label = "Concentration")

        ylims!((0.7, 1.3))
        xlims!((U[1], U[end]))

        #plot!(legend_columns = 2)
        #plot!([0,-1], [-1,-1], linestyle = :solid, color = :white, label = " ")

        #plot!([0,-1], [-1,-1], linestyle = :solid, color = :black, label = "SO")
        #plot!([0,-1], [-1,-1], linestyle = :dash, color = :black, label = "\$W_2\$")

        #scatter!([0,-1], [-1,-1], markershape = :circle, markerstrokecolor = palette(:tab10)[1], color = palette(:tab10)[1], label = "Naïve")
        #scatter!([0,-1], [-1,-1], markershape = :circle, markerstrokecolor = palette(:tab10)[2], color = palette(:tab10)[2], label = "Windowing")
        #scatter!([0,-1], [-1,-1], markershape = :circle, markerstrokecolor = palette(:tab10)[3], color = palette(:tab10)[3], label = "Smoothing")
        #scatter!([0,-1], [-1,-1], markershape = :circle, markerstrokecolor = palette(:tab10)[4], color = palette(:tab10)[4], label = "Concentration")
        #scatter!([0,-1], [-1,-1], markershape = :circle, markerstrokecolor = palette(:tab10)[5], color = palette(:tab10)[5], label = "Intersection")

        #plot!(legend = :topleft)

        display(plt)

        #savefig(plt, "figures/to-discuss-1.pdf")

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