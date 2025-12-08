using Plots, Measures

include("weights.jl")

default() # Reset plot defaults.

gr(size = (275+6+6,183+6).*sqrt(3))
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
        ylabel = "Weight, \$w_t\$",
        xticks = ([0, 25, 50, 75, 100]),
        topmargin = 0pt, 
        rightmargin = 6pt,
        bottommargin = 6pt, 
        leftmargin = 6pt)

ε = 90
ρ = 1
T = 100
P = [1,2,3,4,5]

linewidth = 1
colors = cgrad([palette(:tab10)[1], palette(:tab10)[2]], P[end])
#colors = cgrad([:black, palette(:tab10)[8]], P[end])
#colors = cgrad([:black, :black], P[end])
linestyles = [:solid, :dash, :dashdot, :dashdotdot, :dot]
linewidths = LinRange(1,1.2,P[end])
alpha = 0.117

for p in P
        plot!(1:T, 
                Wp_weights(p, T, ρ/(p*ε)),
                label = "\$p=$p\$",
                color = colors[p],
                linewidth = linewidths[p],
                linestyle = linestyles[p],
                alpha = 1,
                fill = (0, alpha, colors[p]))

end

#xlims!((-2,102))
#yl = ylims(plt)
#ylims!((-0.0007,0.025+0.0007))#yl[2]))

xlims!((-0,100))
yl = ylims(plt)
ylims!((0,yl[2]))

display(plt)

savefig(plt, "figures/weights-for-p.pdf")


# Stop motion talk plots.
if false 
        for q in P

                default() # Reset plot defaults.

                gr(size = (275+6+6,183+6).*sqrt(3))
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
                        ylabel = "Weight, \$w_t\$",
                        xticks = ([0, 25, 50, 75, 100]),
                        topmargin = 0pt, 
                        rightmargin = 6pt,
                        bottommargin = 6pt, 
                        leftmargin = 6pt)

                for p in 1:q
                        plot!(1:T, 
                                Wp_weights(p, T, ρ/(p*ε)),
                                label = "\$p=$p\$",
                                color = colors[p],
                                linewidth = linewidths[p],
                                linestyle = linestyles[p],
                                alpha = 1,
                                fill = (0, alpha, colors[p]))

                end

                #xlims!((-2,102))
                #yl = ylims(plt)
                #ylims!((-0.0007,0.025+0.0007))#yl[2]))

                xlims!((-0,100))
                yl = ylims(plt)
                ylims!((0,yl[2]))

                display(plt)

                savefig(plt, "figures/weights-for-p=1-to-$q.pdf")
        end

end





if false

        LogRange(a, b, n) = exp.(LinRange(log(a), log(b), n))

        plt = plot()

        for s in unique(floor.(Int, LogRange(1,100,30)))
                plot!(1:T, windowing_weights(T, s), label = nothing, color = :black)
        end

        display(plt)

        plt = plot()
        for α in [[0]; LogRange(1e-4,1e0,30)]
                plot!(1:T, smoothing_weights(T, α), label = nothing, color = :black)
        end
        display(plt)

        plt = plot()
        for ρ╱ε in [[0]; LogRange(1e-4,1e0,30)]
                plot!(1:T, Wp_weights(1, T, ρ╱ε), label = nothing, color = :black)
        end
        display(plt)

        plt = plot()
        for ρ╱ε in [[0]; LogRange(1e-4,1e0,30)]
                plot!(1:T, Wp_weights(2, T, ρ╱ε), label = nothing, color = :black)
        end
        display(plt)

        plt = plot(1:T, Wp_weights(1, T, 0), label = nothing, color = :black)
        plot!(1:T, Wp_weights(1, T, 1e-4), label = nothing, color = :black)
        display(plt)

        plt = plot(1:T, Wp_weights(2, T, 0), label = nothing, color = :black)
        plot!(1:T, Wp_weights(2, T, 1e-4), label = nothing, color = :black)
        display(plt)

end



