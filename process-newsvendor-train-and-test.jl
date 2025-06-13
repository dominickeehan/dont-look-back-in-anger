# "C:\Program Files\7-Zip\7z.exe" e "C:\Users\dkee331\Documents\repositories\dont-look-back-in-anger\results\dominic(2).zip" -r -o"C:\Users\dkee331\Documents\repositories\dont-look-back-in-anger\results" *.csv

using CSV, Statistics, StatsBase 
using Plots, Measures

repetitions = 1
number_of_jobs = 1000

U = [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2]

using ProgressBars

function extract_results(skipto, u_index)
    costs = Vector{Union{Missing, Float64}}(undef, number_of_jobs)
    parameter_1s = Vector{Union{Missing, Float64}}(undef, number_of_jobs)
    parameter_2s = Vector{Union{Missing, Float64}}(undef, number_of_jobs)
    objective_values = Vector{Union{Missing, Float64}}(undef, number_of_jobs)

    Threads.@threads for job_number in 0:number_of_jobs-1

        job_index = 999*(u_index-1)+job_number
        results_file = CSV.File("results/$job_index.csv", header = false, skipto = skipto*repetitions)
        try
            costs[job_number+1], parameter_1s[job_number+1], parameter_2s[job_number+1], objective_values[job_number+1] = 
                [[row.Column6, row.Column2, row.Column3, row.Column5] for row in Iterators.take(results_file, repetitions)][1]
        catch
            costs[job_number+1], parameter_1s[job_number+1], parameter_2s[job_number+1], objective_values[job_number+1] = 
                [missing, missing, missing, missing]
        end

        if objective_values[job_number+1] < 0
                #println(job_index)
                #objective_values[job_number+1] = 10000
                #costs[job_number+1] = 10000
        end

    end

    return costs, parameter_1s, parameter_2s, objective_values
end

u_index = 1

naive_SO_results = extract_results(1, u_index)
windowing_SO_results = extract_results(2, u_index)
smoothing_SO_results = extract_results(3, u_index)

naive_W2_results = extract_results(8, u_index)
windowing_W2_results = extract_results(9, u_index)
smoothing_W2_results = extract_results(10, u_index)
concentration_W2_results = extract_results(11, u_index)

intersection_W2_results = extract_results(12, u_index)
argmax(intersection_W2_results[1])
max(intersection_W2_results[1]...)
min(intersection_W2_results[1]...)
max(intersection_W2_results[4]...)
intersection_W2_results[4]

min(skipmissing(intersection_W2_results[4])...)
argmin(skipmissing(intersection_W2_results[4]))
#println(intersection_W2_results[4])

function display_extracted_results(name, extracted_results)
    μ = mean(skipmissing(extracted_results[1]))
    σ = sem(skipmissing(extracted_results[1]))

    println(name*": $μ ± $σ")

    display(sort(collect(pairs(countmap(eachrow(hcat(extracted_results[2], extracted_results[3]))))), by = x->x.second, rev = true))

    display(count(ismissing, extracted_results[1]))

end

#display_extracted_results("Naive W1", naive_SO_results)
#display_extracted_results("Windowing W1", windowing_SO_results)
#display_extracted_results("Smoothing W1", smoothing_SO_results)

#display_extracted_results("Naive W2", naive_W2_results)
#display_extracted_results("Windowing W2", windowing_W2_results)
#display_extracted_results("Smoothing W2", smoothing_W2_results)
display_extracted_results("Concentration W2", concentration_W2_results)

display_extracted_results("Intersection W2", intersection_W2_results)


function extract_line_to_plot(skipto)

    expected_costs = zeros(length(U))
    sems = zeros(length(U))

    for u_index in eachindex(U)

        error_indices = Int[]

        for i in [1,2,3,8,9,10,11,12]
                #_, _, _, values = extract_results(i, u_index)
                #push!(error_indices, findall(<=(0.0), values)...)
        end
        
        costs, _, _, _ = extract_results(skipto, u_index)

        costs[error_indices] .= missing

        expected_costs[u_index] = mean(skipmissing(costs))
        sems[u_index] = sem(skipmissing(costs))

    end

    return expected_costs, sems

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
            xlabel = "Extent of shift, \$u\$", 
            ylabel = "Expected cost (normalized)",)

fillalpha = 0.1

normalizer, _ = extract_line_to_plot(3)

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
expected_costs, sems = extract_line_to_plot(3)
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
xlims!((0.0001, 0.01))

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








function extract_histogram_to_plot(skipto, u_index)

        expected_costs, _, _, objective_values = extract_results(skipto, u_index)

        return 100*(expected_costs - objective_values)./(objective_values)

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

plt=plot(xlims=(-100,300), ylabel="Frequency (normalized)", xlabel="Out-of-sample disappointment (%)")
u_index = 9
#stephist!(extract_histogram_to_plot(9, u_index), color=palette(:tab10)[2], normalize=:pdf, label="Windowing",)
#stephist!(extract_histogram_to_plot(10, u_index), color=palette(:tab10)[3], normalize=:pdf, fill=true, fillalpha=0.1, label="\$W_2\$ Smoothing")
stephist!(extract_histogram_to_plot(11, u_index), color=palette(:tab10)[4], linestyle=:dash, normalize=:pdf, fill=true, fillalpha=0.1, label="\$W_2\$ Concentration")
#vline!([mean(extract_histogram_to_plot(9, u_index))], color=palette(:tab10)[2], label=nothing)
#vline!([mean(extract_histogram_to_plot(10, u_index))], color=palette(:tab10)[3], label=nothing)
vspan!([-sem(extract_histogram_to_plot(11, u_index))+mean(extract_histogram_to_plot(11, u_index)),sem(extract_histogram_to_plot(11, u_index))+mean(extract_histogram_to_plot(11, u_index))], color=palette(:tab10)[4], alpha=0.1, label=nothing)
vline!([mean(extract_histogram_to_plot(11, u_index))], color=palette(:tab10)[4], linestyle=:dash, label=nothing)
stephist!(extract_histogram_to_plot(12, u_index), color=palette(:tab10)[5], linestyle=:dash, normalize=:pdf, fill=true, fillalpha=0.1, label="\$W_2\$ Intersections")
vspan!([-sem(extract_histogram_to_plot(12, u_index))+mean(extract_histogram_to_plot(12, u_index)),sem(extract_histogram_to_plot(12, u_index))+mean(extract_histogram_to_plot(12, u_index))], color=palette(:tab10)[5], alpha=0.1, label=nothing)
vline!([mean(skipmissing(extract_histogram_to_plot(12, u_index)))], color=palette(:tab10)[5], linestyle=:dash, label=nothing)
display(plt)

#x = extract_histogram_to_plot(12, u_index)
#min(x...)