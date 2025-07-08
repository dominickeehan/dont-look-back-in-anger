using JuMP, Ipopt
using Plots, Measures

include("weights.jl")

T = 100

GeomRange(a, b, n) = exp.(LinRange(log(a), log(b), n))

ϱ╱ε = [[0]; GeomRange(1e-4,1e0,40)]

#=
plt = plot()
for i in ϱ╱ε
        plot!(1:T, W1_concentration_weights(T, i), label = nothing, color = :black)
end
display(plt)

plt = plot()
for i in ϱ╱ε
        plot!(1:T, W2_concentration_weights(T, i), label = nothing, color = :black)
end
display(plt)

plt = plot()
for i in [[0]; GeomRange(1e-4,1e0,40)]
#        plot!(1:T, smoothing_weights(T, i), label = nothing, color = :black)
end
display(plt)

plt = plot()
for i in [round.(Int, LinRange(1,10,10)); round.(Int, LinRange(12,30,10)); round.(Int, LinRange(33,60,10)); round.(Int, LinRange(64,100,10));]
#        plot!(1:T, windowing_weights(T, i), label = nothing, color = :black)
end
display(plt)

=#

ε = 75

ϱ = 1

linewidth = 1

default() # Reset plot defaults.

gr(size = (sqrt(6/10)*600,sqrt(6/10)*400))

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
        xminorticks = 0, 
        yminorticks = 0,
        fontfamily = font_family,
        guidefont = primary_font,
        tickfont = secondary_font,
        legendfont = legend_font)

plt = plot(
            xlabel = "Time index, \$t\$", 
            ylabel = "Weight, \$w_t\$",
            xticks = ([0, 25, 50, 75, 100]),
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 3pt, 
            leftmargin = 3pt)

linestyles = [:solid, :dash, :dashdot]

tab10_primary_colour = [188, 189, 34]/256

for p in 1:3

    plot!(1:T, 
            Wp_concentration_weights(p, T, ϱ/ε), 
            label = "\$p=$p\$",
            color = RGB(tab10_primary_colour[1]-0.0*(p-1),tab10_primary_colour[2]-0.0*(p-1),max(tab10_primary_colour[3]-0.0*(p-1),0)),
            #markershape = :circle,
            #markersize = 2,
            #markerstrokewidth = 1,
            #markerstrokecolor = :black,
            linewidth = linewidth,
            linestyle = linestyles[p],
            alpha = 1)

end

display(plt)
yl = ylims(plt)

savefig(plt, "figures/optimal-weights-for-varying-p.pdf")



include("weights.jl")

default() # Reset plot defaults.

gr(size = (sqrt(6/10)*600,sqrt(6/10)*400))

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
        xminorticks = 0, 
        yminorticks = 0,
        fontfamily = font_family,
        guidefont = primary_font,
        tickfont = secondary_font,
        legendfont = legend_font)

plt = plot(
            xlabel = "Time index, \$t\$", 
            ylabel = "Weight, \$w_t\$",
            xticks = ([0, 25, 50, 75, 100]),
            legend = :topleft,
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 3pt, 
            leftmargin = 3pt)

plot!(1:T, 
        smoothing_weights(T, 3*ϱ/(ε+ϱ)), 
        label = "Smoothing",
        color = palette(:tab10)[2],
        linewidth = linewidth,
        linestyle = :solid,
        alpha = 1)

plot!(1:T, 
        W1_concentration_weights(T, ϱ/ε), 
        label = "Concentration",
        color = palette(:tab10)[9],
        linewidth = linewidth,
        linestyle = :dash,
        alpha = 1)



        
ylims!(yl)
display(plt)

savefig(plt, "figures/optimal-weights-and-smoothing-weights.pdf")





