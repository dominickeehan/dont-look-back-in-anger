using JuMP, Ipopt
using Plots, Measures

T = 100

function concentration_weights(ϱ, ε, p)

    if ϱ >= (1/1)*ε; ϱ = (1/1)*ε; end

    Problem = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

    @variable(Problem, 1>= w[t=1:T] >=0)

    @constraint(Problem, sum(w[t] for t in 1:T) == 1)
    @constraint(Problem, sum(w[t]*t^p*ϱ^p for t in 1:T) <= ε^p)

    @objective(Problem, Max, (1/(sum(w[t]^2 for t in 1:T)))*((ε-(sum(w[t]*t^p*ϱ^p for t in 1:T))^(1/p))^(2*p)))

    optimize!(Problem)

    weights = [max(value(w[t]),0) for t in 1:T]
    weights .= weights/sum(weights)

    return reverse(weights)

end

ϱ = 1
ε = 75

default() # Reset plot defaults.

gr(size = (sqrt(7/10)*600,sqrt(7/10)*400))

font_family = "Computer Modern"
primary_font = Plots.font(font_family, pointsize = 17)
secondary_font = Plots.font(font_family, pointsize = 11)
legend_font = Plots.font(font_family, pointsize = 15)

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
        fontfamily = font_family,
        guidefont = primary_font,
        tickfont = secondary_font,
        legendfont = legend_font)

plt = plot(
            xlabel = "\$t\$", 
            ylabel = "Datum weight",
            xticks = ([1, 25, 50, 75, 100]),
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 3pt, 
            leftmargin = 3pt)

linestyles = [:solid, :dash, :dashdot]

for p in 1:3

    plot!(1:T, 
            concentration_weights(ϱ, ε, p), 
            label = "\$p=$p\$",
            color = palette(:tab10)[p],
            #markershape = :circle,
            #markersize = 2,
            #markerstrokewidth = 1,
            #markerstrokecolor = :black,
            linewidth = 2,
            linestyle = linestyles[p],
            alpha = 1)

end

display(plt)
yl = ylims(plt)

savefig(plt, "figures/optimal-weights-for-varying-p.pdf")



include("weights.jl")

default() # Reset plot defaults.

gr(size = (sqrt(7/10)*600,sqrt(7/10)*400))

font_family = "Computer Modern"
primary_font = Plots.font(font_family, pointsize = 17)
secondary_font = Plots.font(font_family, pointsize = 11)
legend_font = Plots.font(font_family, pointsize = 15)

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
        fontfamily = font_family,
        guidefont = primary_font,
        tickfont = secondary_font,
        legendfont = legend_font)

plt = plot(
            xlabel = "\$t\$", 
            ylabel = "Datum weight",
            xticks = ([1, 25, 50, 75, 100]),
            legend = :topleft,
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 3pt, 
            leftmargin = 3pt)

plot!(1:T, 
        concentration_weights(ϱ, ε, 1), 
        label = "Optimal",
        color = palette(:tab10)[1],
        linewidth = 2,
        linestyle = :solid,
        alpha = 1)

plot!(1:T, 
        smoothing_weights(T, ε, 3*ϱ/(ε+ϱ)), 
        label = "Smoothing",
        color = palette(:tab10)[4],
        linewidth = 2,
        linestyle = :dashdotdot,
        alpha = 1)

        
ylims!(yl)
display(plt)

savefig(plt, "figures/optimal-weights-and-smoothing-weights.pdf")