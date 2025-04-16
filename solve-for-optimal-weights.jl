using JuMP, Ipopt
using Plots, Measures

T = 100
p = 2

d = [t for t in 1:T]
#d = [t-2 for t in 1:T]
#d[1] = 1

function solve_for_weights(ϵ, ρ)

    Problem = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "tol" => 1e-9))

    @variable(Problem, 1>= w[t=1:T] >=0)

    @constraint(Problem, sum(w[t] for t in 1:T) == 1)
    @constraint(Problem, (sum(w[t]*t^p for t in 1:T)*ρ^p)^(1/p) <= ϵ)
    for t in 1:T-1; @constraint(Problem, w[t] >= w[t+1]); end

    @objective(Problem, Max, (1/(sum(w[t]^2 for t in 1:T)))*((ϵ-(sum(w[t]*d[t]^p for t in 1:T)*ρ^p)^(1/p))^(2*p)))

    optimize!(Problem)

    display(objective_value(Problem))

    weights = [value(w[t]) for t in 1:T]

    display(weights)

    default() # Reset plot defaults.

    gr(size = (600,400))
    
    font_family = "Computer Modern"
    primary_font = Plots.font(font_family, pointsize = 17)
    secondary_font = Plots.font(font_family, pointsize = 11)
    legend_font = Plots.font(font_family, pointsize = 16)
    
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
    
    plt = plot(1:T, weights, 
            xlabel = "\$t\$", 
            ylabel = "\$w_t\$",
            legend = nothing,
            color = palette(:tab10)[1],
            alpha = 1,
            linewidth = 1,
            fillalpha = .1,
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 3pt, 
            leftmargin = 3pt)
    
    display(plt);

    #return reverse(weights)

    #savefig(plt, "figures/W1-weights.pdf")
    #savefig(plt, "figures/W2-weights.pdf")

end

#solve_for_weights(10.0,0.5)
#solve_for_weights(10.0,0.1)

solve_for_weights(100,0.1)