using Plots, Measures

include("weights.jl")

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

plt = plot(
        xlabel = "Time index, \$t\$", 
        ylabel = "Observation weight, \$w_t\$",
        xticks = ([0, 25, 50, 75, 100]),
        topmargin = 0pt, 
        rightmargin = 0pt,
        bottommargin = 6pt, 
        leftmargin = 6pt)

ε = 100
ρ = 1
T = 100
P = [1,2,3,4,5]

linewidth = 1
colors = cgrad([palette(:tab10)[1], palette(:tab10)[2]], P[end])
linestyles = [:solid, :dash, :dashdot, :dashdotdot, :dot]
alpha = 0.05

for p in P
        plot!(1:T, 
                Wp_weights(p, T, ρ/(p*ε)),
                label = "\$p=$p\$",
                color = colors[p],
                linewidth = linewidth,
                linestyle = linestyles[p],
                alpha = 1,
                fill = (0, alpha, colors[p]))

end

xlims!((-2,102))
yl = ylims(plt)
ylims!((0,yl[2]))
display(plt)

savefig(plt, "figures/weights-for-p.pdf")








if false

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
        plot!(1:T, W1_weights(T, i), label = nothing, color = :black)
        end
        display(plt)

        plt = plot()
        for i in ρ╱ε
        plot!(1:T, W2_weights(T, i), label = nothing, color = :black)
        end
        display(plt)

        #plt = plot(1:T, W1_weights(T, 0), label = nothing, color = :black)
        #plot!(1:T, W1_weights(T, 1e-4), label = nothing, color = :black)
        #display(plt)

        #plt = plot(1:T, W2_weights(T, 0), label = nothing, color = :black)
        #plot!(1:T, W2_weights(T, 1e-4), label = nothing, color = :black)
        #display(plt)

end
