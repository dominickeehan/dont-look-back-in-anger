using Random, Statistics, StatsBase, Distributions
using ProgressBars

Random.seed!(42)

repetitions = 100
history_length = 100

initial_demand_probability = 0.1 # 0.1

U = [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2, 5e-2]

u = 1e-3

shift_distribution = Uniform(-u,u)

demand_sequences = [zeros(history_length) for _ in 1:repetitions]

for repetition in 1:repetitions
    local demand_probability = initial_demand_probability

    for t in 1:history_length
        demand_sequences[repetition][t] = rand(Binomial(10000, demand_probability))  
        demand_probability = min(max(demand_probability + rand(shift_distribution), 0), 1)

    end
end



using Plots, Measures

default() # Reset plot defaults.

gr(size = (317+6,159+6).*sqrt(3))

fontfamily = "Computer Modern"
default(framestyle = :box,
        grid = true,
        #gridlinewidth = 1.0,
        gridalpha = 0.075,
        #minorgrid = true,
        #minorgridlinewidth = 1.0, 
        #minorgridalpha = 0.075,
        #minorgridlinestyle = :dash,
        tick_direction = :in,
        xminorticks = 0, 
        yminorticks = 0,
        fontfamily = fontfamily,
        guidefont = Plots.font(fontfamily, pointsize = 12),
        legendfont = Plots.font(fontfamily, pointsize = 11),
        tickfont = Plots.font(fontfamily, pointsize = 10),)

plt = plot(1:history_length, 
        stack(demand_sequences[2:repetitions]), 
        xlims = (-2,history_length+2),
        xticks = ([0, 25, 50, 75, 100]),
        xlabel = "Time, \$t\$", 
        ylabel = "Demand, \$ξ_t\$",
        labels = nothing, 
        #linecolor = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5]],
        #markercolor = palette(:tab10)[1],
        #markershape = :circle,
        color = palette(:tab10)[8],
        alpha = 0.05,
        #linestyle = :auto,
        #markersize = 4, 
        #markerstrokewidth = 1,
        #markerstrokecolor = :black,
        topmargin = 0pt, 
        rightmargin = 0pt,
        bottommargin = 6pt, 
        leftmargin = 6pt,
        )

plot!(1:history_length, 
        stack(demand_sequences[1]), 
        labels = nothing, 
        #linecolor = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5]],
        markercolor = palette(:tab10)[8],
        markershape = :circle,
        color = palette(:tab10)[8],
        alpha = 1.0,
        #linestyle = :auto,
        markersize = 3, 
        markerstrokewidth = 1,
        markerstrokecolor = :black,
        )

display(plt)

savefig(plt, "figures/demand_sequences.pdf")
