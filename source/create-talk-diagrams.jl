using Random, Distributions, Statistics, StatsBase

# Draws the distribution diagrams for the talk.

Prob = Normal(0,1)
Qrob = MixtureModel(Normal,  [(-0.75, 1), (1.75, 0.5)], [0.8, 0.2])

using Plots, Measures



# Distribution P.
default() # Reset plot settings to defaults.

gr(size = (275,183).*sqrt(3))

fontfamily = "Computer Modern"

default(framestyle = :none,
        grid = false,
        minorgrid = false,
        background_color = :transparent,
        ytick_direction = :none,
        xtick_direction = :in,
        xminorticks = 0,
        yminorticks = 0,
        fontfamily = fontfamily,
        guidefont = Plots.font(fontfamily, pointsize = 12),
        tickfont = Plots.font(fontfamily, pointsize = 10),
        legendfont = Plots.font(fontfamily, pointsize = 11))

domain = LinRange(-3, 3, 1000)
yl = (domain[1],domain[end])


plt = plot(xlabel = nothing,
           ylabel = nothing,
           topmargin = 0pt,
           rightmargin = 0pt,
           bottommargin = 0pt,
           leftmargin = 0pt,)

plot!(domain,
      [pdf(Prob, ξ) for ξ in domain],
      color = palette(:tab10)[1],
      linewidth = 3,
      fill = (0, 0.1, palette(:tab10)[1]),
      label = nothing)

display(plt)
savefig(plt,"figures/prob.svg")



# Distribution Q.
default() # Reset plot settings to defaults.

gr(size = (275,183).*sqrt(3))

fontfamily = "Computer Modern"

default(framestyle = :none,
        grid = false,
        minorgrid = false,
        background_color = :transparent,
        ytick_direction = :none,
        xtick_direction = :in,
        xminorticks = 0,
        yminorticks = 0,
        fontfamily = fontfamily,
        guidefont = Plots.font(fontfamily, pointsize = 12),
        tickfont = Plots.font(fontfamily, pointsize = 10),
        legendfont = Plots.font(fontfamily, pointsize = 11))

domain = LinRange(-3, 3, 1000)
yl = (domain[1],domain[end])


plt = plot(xlabel = nothing,
           ylabel = nothing,
           topmargin = 0pt,
           rightmargin = 0pt,
           bottommargin = 0pt,
           leftmargin = 0pt,)

plot!(domain,
      [pdf(Qrob, ξ) for ξ in domain],
      color = palette(:tab10)[2],
      linewidth = 3,
      fill = (0, 0.1, palette(:tab10)[2]),
      label = nothing)

display(plt)
savefig(plt,"figures/qrob.svg")



# Five identical distributions with a demand drawn from each.
Prob = [Normal(0,1), Normal(0,1), Normal(0,1), Normal(0,1), Normal(0,1)]

demands = [-1, 0.1, 2, -0.25, 0.75]

using Plots, Measures

default() # Reset plot settings to defaults.

gr(size = (337+6,224+6).*sqrt(3))

fontfamily = "Computer Modern"

default(framestyle = :none,
        grid = false,
        minorgrid = false,
        background_color = :transparent,
        ytick_direction = :none,
        xtick_direction = :in,
        xminorticks = 0,
        yminorticks = 0,
        fontfamily = fontfamily,
        guidefont = Plots.font(fontfamily, pointsize = 12),
        tickfont = Plots.font(fontfamily, pointsize = 10),
        legendfont = Plots.font(fontfamily, pointsize = 11))

domain = LinRange(-5, 5, 1000)
yl = (domain[1],domain[end])

horizontal_increment = 0.25

plt = plot(xlabel = nothing,
           ylabel = nothing,
           topmargin = 0pt,
           rightmargin = 0pt,
           bottommargin = 0pt,
           leftmargin = 0pt,)

for t in 1:5

        alpha = 0.1

        plot!([-pdf(Prob[t], ξ) for ξ in domain].+horizontal_increment*t,
              domain,
              color = palette(:tab10)[1],
              linewidth = 1.5,
              fill = (0, alpha, palette(:tab10)[1]),
              label = nothing)

        scatter!([-pdf(Prob[t], demands[t])+horizontal_increment*t],
                 [demands[t]],
                 markershape = :circle,
                 markersize = 5,
                 markerstrokecolor = palette(:tab10)[2],
                 markerstrokewidth = 0.0,
                 markercolor = palette(:tab10)[2],
                 alpha = 1,
                 label = nothing)

end

display(plt)
savefig(plt,"figures/probs.svg")



# Five drifting distributions with a demand drawn from each.
Prob = [Normal(0,1), Normal(1,1), Normal(0.5,1.5), MixtureModel(Normal,  [(-0.75, 1), (1.75, 0.5)], [0.8, 0.2]), MixtureModel(Normal,  [(-2, 1), (2, 1)], [0.5, 0.5])]

demands = [-1, 1.1, 2.5, -2, 3]

using Plots, Measures

default() # Reset plot settings to defaults.

gr(size = (337+6,224+6).*sqrt(3))

fontfamily = "Computer Modern"

default(framestyle = :none,
        grid = false,
        minorgrid = false,
        background_color = :transparent,
        ytick_direction = :none,
        xtick_direction = :in,
        xminorticks = 0,
        yminorticks = 0,
        fontfamily = fontfamily,
        guidefont = Plots.font(fontfamily, pointsize = 12),
        tickfont = Plots.font(fontfamily, pointsize = 10),
        legendfont = Plots.font(fontfamily, pointsize = 11))

domain = LinRange(-5, 5, 1000)
yl = (domain[1],domain[end])

horizontal_increment = 0.25

plt = plot(xlabel = nothing,
           ylabel = nothing,
           topmargin = 0pt,
           rightmargin = 0pt,
           bottommargin = 0pt,
           leftmargin = 0pt,)

for t in 1:5

        alpha = 0.1

        plot!([-pdf(Prob[t], ξ) for ξ in domain].+horizontal_increment*t,
              domain,
              color = palette(:tab10)[1],
              linewidth = 1.5,
              fill = (0, alpha, palette(:tab10)[1]),
              label = nothing)

        scatter!([-pdf(Prob[t], demands[t])+horizontal_increment*t],
                 [demands[t]],
                 markershape = :circle,
                 markersize = 5,
                 markerstrokecolor = palette(:tab10)[2],
                 markerstrokewidth = 0.0,
                 markercolor = palette(:tab10)[2],
                 alpha = 1,
                 label = nothing)

end

display(plt)
savefig(plt,"figures/qrobs.svg")
