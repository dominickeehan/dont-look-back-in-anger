using Plots, Measures

include("weights.jl")

T = 100

ε = 75
ρ = 1


default() # Reset plot defaults.

gr(size = (275+6,183+6).*sqrt(3))

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
        tickfont = Plots.font(fontfamily, pointsize = 10))

linewidth = 1

plt = plot(
            xlabel = "Time index, \$t\$", 
            ylabel = "Weight, \$w_t\$",
            xticks = ([0, 25, 50, 75, 100]),
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 6pt, 
            leftmargin = 6pt)

true_objective_value(t, ε, ρ) = t * ((max(ε - (0.5)*(t+1)*ρ, 0)^2.0))

function formula_for_optimal_window_size(T, ε, ρ)
    unprojected_formula = (1.0/(3.0)) * (2.0*(ε/ρ) - 1.0)
    
    projected_floor_formula = max(min(floor(unprojected_formula), T), 1)
    objective_projected_floor_formula = true_objective_value(projected_floor_formula, ε, ρ)

    projected_ceil_formula = max(min(ceil(unprojected_formula), T), 1)
    objective_projected_ceil_formula = true_objective_value(projected_ceil_formula, ε, ρ)

    if objective_projected_floor_formula >= objective_projected_ceil_formula

        return projected_floor_formula
    else

        return projected_ceil_formula
    end
end

plot!(1:T, 
        windowing_weights(T, round(Int, formula_for_optimal_window_size(T, ε, ρ))),
        seriestype = :steppre,
        label = "Windowing",
        color = palette(:tab10)[1],
        linewidth = linewidth,
        linestyle = :solid,
        alpha = 1,
        fill = (0, 0.1, palette(:tab10)[1]))

plot!(1:T, 
        smoothing_weights(T, 3/((ε/ρ)+1)), 
        label = "Smoothing",
        color = palette(:tab10)[7],
        linewidth = linewidth,
        linestyle = :dash,
        alpha = 1,
        fill = (0, 0.1, palette(:tab10)[7]))

plot!(1:T, 
        W1_concentration_weights(T, ρ/ε), 
        label = "Optimal",
        color = palette(:tab10)[9],
        linewidth = linewidth,
        linestyle = :dashdot,
        alpha = 1,
        fill = (0, 0.1, palette(:tab10)[9]))

xlims!((-2,102))
yl = ylims(plt)
ylims!((0,yl[2]))
#display(plt)

#savefig(plt, "figures/weights-for-p=1.pdf")








default() # Reset plot defaults.

gr(size = (275+6,183+6).*sqrt(3))

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
        tickfont = Plots.font(fontfamily, pointsize = 10))

linewidth = 1

plt = plot(
            xlabel = "Time index, \$t\$", 
            ylabel = "Weight, \$w_t\$",
            xticks = ([0, 25, 50, 75, 100]),
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 6pt, 
            leftmargin = 6pt)

ε = 100
ρ = 1

plot!(1:T, 
        Wp_concentration_weights(1, T, ρ/(ε)),#W2_concentration_weights(T, ρ/(2*ε)), 
        label = "\$p=1\$",
        color = palette(:tab10)[1],
        linewidth = linewidth,
        linestyle = :solid,
        alpha = 1,
        fill = (0, 0.1, palette(:tab10)[1]))

plot!(1:T, 
        Wp_concentration_weights(2, T, ρ/(2*ε)),#W2_concentration_weights(T, ρ/(2*ε)), 
        label = "\$p=2\$",
        color = palette(:tab10)[2],
        linewidth = linewidth,
        linestyle = :dash,
        alpha = 1,
        fill = (0, 0.1, palette(:tab10)[2]))

plot!(1:T, 
        Wp_concentration_weights(3, T, ρ/(3*ε)), 
        label = "\$p=3\$",
        color = palette(:tab10)[3],
        linewidth = linewidth,
        linestyle = :dashdot,
        alpha = 1,
        fill = (0, 0.1, palette(:tab10)[3]))

plot!(1:T, 
        Wp_concentration_weights(4, T, ρ/(4*ε)), 
        label = "\$p=4\$",
        color = palette(:tab10)[4],
        linewidth = linewidth,
        linestyle = :dashdotdot,
        alpha = 1,
        fill = (0, 0.1, palette(:tab10)[4]))

plot!(1:T, 
        Wp_concentration_weights(5, T, ρ/(5*ε)), 
        label = "\$p=5\$",
        color = palette(:tab10)[5],
        linewidth = linewidth,
        linestyle = :dot,
        alpha = 1,
        fill = (0, 0.1, palette(:tab10)[5]))

xlims!((-2,102))
yl = ylims(plt)
ylims!((0,yl[2]))
display(plt)
#yl = ylims(plt)

savefig(plt, "figures/weights-for-p=1,2,3,4,5.pdf")









LogRange(a, b, n) = exp.(LinRange(log(a), log(b), n))

plt = plot()
#for i in [round.(Int, LinRange(1,10,10)); round.(Int, LinRange(12,30,10)); round.(Int, LinRange(33,60,10)); round.(Int, LinRange(64,100,10));]
for i in [unique(round.(Int, LogRange(1,100,40)));]
        plot!(1:T, windowing_weights(T, i), label = nothing, color = :black)
end
display(plt)

plt = plot()
for i in [[0]; LogRange(1e-4,1e0,40)]
    plot!(1:T, smoothing_weights(T, i), label = nothing, color = :black)
end
display(plt)


ρ╱ε = [[0]; LogRange(1e-4,1e0,40)]

plt = plot()
for i in ρ╱ε
    plot!(1:T, W1_concentration_weights(T, i), label = nothing, color = :black)
end
display(plt)

plt = plot()
for i in ρ╱ε
    plot!(1:T, W2_concentration_weights(T, i), label = nothing, color = :black)
end
display(plt)

#plt = plot(1:T, W1_concentration_weights(T, 0), label = nothing, color = :black)
#plot!(1:T, W1_concentration_weights(T, 1e-4), label = nothing, color = :black)
#display(plt)

#plt = plot(1:T, W2_concentration_weights(T, 0), label = nothing, color = :black)
#plot!(1:T, W2_concentration_weights(T, 1e-4), label = nothing, color = :black)
#display(plt)


