using Random, Distributions, Statistics, StatsBase
using JuMP, Gurobi
using Plots, Measures
using ProgressBars




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


plt = plot(xlims=(-4,4),
           ylims=(0,3),
           xlabel="Mean", 
           ylabel="Standard deviation",
           topmargin = 0pt, 
           rightmargin = 0pt,
           bottommargin = 6pt, 
           leftmargin = 6pt)

function nonnegative_ellipse_coords(horizontal_radius, vertical_radius, x_centre, y_centre)

    t = range(1.5*π, -0.5*π; length=800)

    x = x_centre .+ horizontal_radius .* cos.(t)
    y = y_centre .+ vertical_radius .* sin.(t)
    
    negative_y_indices = y .< 0
    y[negative_y_indices] .= 0

    return x, y

end



    linewidth = 1
    alpha = 1
    fillalpha = 0.2

    x_coords, y_coords = nonnegative_ellipse_coords(2.7,2.5,-1,0.1)
    plot!(x_coords,
          y_coords,
          color = palette(:tab10)[1],
          linewidth = 0,
          linestyle = :solid,
          alpha = 0,
          label = nothing,
          fill = (0, fillalpha, palette(:tab10)[1]))

    
    x_coords, y_coords = nonnegative_ellipse_coords(2.2,2.1,0,0)
    plot!(x_coords,
          y_coords,
          color = palette(:tab10)[1],
          linewidth = 0,
          linestyle = :solid,
          alpha = 0,
          label = nothing,
          fill = (0, fillalpha, palette(:tab10)[1]))

    x_coords, y_coords = nonnegative_ellipse_coords(1.5,1.5,1,0.1)
    plot!(x_coords,
          y_coords,
          color = palette(:tab10)[1],
          linewidth = 0,
          linestyle = :solid,
          alpha = 0,
          label = nothing,
          fill = (0, fillalpha, palette(:tab10)[1]))

function nonnegative_intersected_ellipse_coords(horizontal_radii, vertical_radii, x_centres, y_centres)

    x_coords, y_coords = nonnegative_ellipse_coords(horizontal_radii[1],vertical_radii[1],x_centres[1],y_centres[1]) 

    for i in eachindex(x_coords)
        if i ∈ [1,200]∪[601,800] # Negative quadrant so take negative root.
            try
                y_coords[i] = min(y_coords[i], y_centres[2] - vertical_radii[2]*sqrt(1-((x_coords[i]-x_centres[2])/horizontal_radii[2])^2))

            catch # If outside shared x domain.
                y_coords[i] = -99

            end

        else # Positive quadrant so take positive root.
            try
                y_coords[i] = min(y_coords[i], y_centres[2] + vertical_radii[2]*sqrt(1-((x_coords[i]-x_centres[2])/horizontal_radii[2])^2))

            catch # If outside shared x domain.
                y_coords[i] = -99

            end
        end
    end

    in_shared_x_domain_y_coords_indices = y_coords .!= -99
    y_coords = y_coords[in_shared_x_domain_y_coords_indices]
    x_coords = x_coords[in_shared_x_domain_y_coords_indices]

    negative_y_coords_indices = y_coords .< 0
    y_coords[negative_y_coords_indices] .= 0

    return x_coords, y_coords

end

    #x_coords, y_coords = nonnegative_intersected_ellipse_coords([2.7,1.5],[2.5,1.5],[-1,1],[0.1,0.1])
    x_coords, y_coords = nonnegative_intersected_ellipse_coords([1.5,2.7],[1.5,2.5],[1,-1],[0.1,0.1])

    plot!(x_coords,
          y_coords,
          color = palette(:tab10)[1],
          linewidth = linewidth,
          linestyle = :solid,
          alpha = 1,
          label = nothing)

    x_coords, y_coords = nonnegative_ellipse_coords(1,1,-1,-1)
    plot!(x_coords,
          y_coords,
          color = palette(:tab10)[1],
          linewidth = linewidth,
          linestyle = :solid,
          alpha = alpha,
          label = "Intersections",
          fill = (0, 2*fillalpha, palette(:tab10)[1]))

    x_coords, y_coords = nonnegative_ellipse_coords(1.1,1,1,0.1)
    plot!(x_coords,
          y_coords,
          color = palette(:tab10)[2],
          linewidth = linewidth,
          linestyle = :dash,
          alpha = alpha,
          label = "Weighted",
          fill = (0, 2*fillalpha, palette(:tab10)[2]))


    #scatter!([-1], [-1], color=RGB(tab10_primary_colour[1]-0.1,tab10_primary_colour[2]-0.1,tab10_primary_colour[3]-0.1), markersize=markersize, markerstrokewidth=0.0, alpha=1, label="Intersections",)

    tab10_primary_colour = [188, 189, 34]/255

    #scatter!([-1], [-1], color=RGB(tab10_primary_colour[1],tab10_primary_colour[2],tab10_primary_colour[3]), markersize=markersize, markerstrokewidth=0.0, alpha=1, label="Concentration",)

    scatter!([-1,0,1], [0,0,0], 
                markersize = 6.0,
                markershape = :utriangle,
                markercolor = :black,#palette(:tab10)[8],
                markerstrokecolor = :black,
                markerstrokewidth = 0,#,0.5,
                alpha=1, labels=nothing,)
    annotate!(-1, 0, text(" \$\\xi_1\$", :black, :bottom, 12))
    annotate!(0, 0, text(" \$\\xi_2\$", :black, :bottom, 12))
    annotate!(1, 0, text(" \$\\xi_3\$", :black, :bottom, 12))

    display(plt)
    savefig(plt, "figures/ambiguity-sets.pdf")


throw=throw






Random.seed!(42)

env = Gurobi.Env()
GRBsetintparam(env, "OutputFlag", 0)
Linear_Optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

function in_W2_ball(P, ε, Q)

    # P, Q := [points, weights] .

    m = length(P[1])
    n = length(Q[1])

    d = [abs(ξ - ζ)^2 for ξ in P[1], ζ in Q[1]]

    model = Model(Linear_Optimizer)

    @variable(model, γ[1:m, 1:n] >= 0)

    @constraint(model, [i=1:m], sum(γ[i,j] for j in 1:n) == P[2][i])
    @constraint(model, [j=1:n], sum(γ[i,j] for i in 1:m) == Q[2][j])

    @objective(model, Min, sum(γ[i,j] * d[i,j] for i in 1:m, j in 1:n))

    optimize!(model)

    try; return ifelse(sqrt(objective_value(model)) <= ε, 1, 0); catch; return 0; end

end

include("weights.jl")

normalise(x) = x/sum(x)

function mean_and_std(Q)

    return mean(Q[1], Weights(Q[2])), std(Q[1], Weights(Q[2]))

end

support = [-4,4]
number_of_points = 10

number_of_distributions = 2000 # 30000

markersize = 10

ε = 1

for ϱ in [ε/2]

    P = [[-1,0,1], W2_concentration_weights(3, ϱ/ε)]

    Qs = [[zeros(number_of_points), zeros(number_of_points)] for _ in 1:number_of_distributions]
    Threads.@threads for i in ProgressBar(eachindex(Qs))
        local n = number_of_points
        local μ = rand(Uniform(support[1],support[end]))
        local σ = rand(Uniform(0,3))
        local points = rand(Normal(μ, σ), n)

        Qs[i] = [points, 1/n*ones(n)]

    end

    plot_intersection_Qs = [zeros(3) for _ in eachindex(Qs)]
    Threads.@threads for i in ProgressBar(eachindex(Qs))
        plot_intersection_Qs[i] = [in_W2_ball([[P[1][j]],[1.0]],ε+(3-j+1)*ϱ,Qs[i]) for j in 1:3]

    end

    default() # Reset plot defaults.

    gr(size = (600,400))

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

    plt = plot(xlims=(support[1],support[end]), ylims=(0,3), xlabel="Mean", ylabel="Standard deviation")

    tab10_primary_colour = [227,119,194]/255

    for i in ProgressBar(eachindex(Qs))

        if sum(plot_intersection_Qs[i]) == 1
            mean, std = mean_and_std(Qs[i])
            scatter!([mean], [std], color=RGB(min(tab10_primary_colour[1]+0.1,1.0),tab10_primary_colour[2]+0.1,tab10_primary_colour[3]+0.1), markersize=markersize, markerstrokewidth=0.0, alpha=1, labels=nothing,)

        end
    end

    for i in ProgressBar(eachindex(Qs))
        if sum(plot_intersection_Qs[i]) == 2
            mean, std = mean_and_std(Qs[i])
            scatter!([mean], [std], color=RGB(tab10_primary_colour[1],tab10_primary_colour[2],tab10_primary_colour[3]), markersize=markersize, markerstrokewidth=0.0, alpha=1, labels=nothing,)

        end
    end

    for i in ProgressBar(eachindex(Qs))
        if sum(plot_intersection_Qs[i]) == 3
            mean, std = mean_and_std(Qs[i])
            scatter!([mean], [std], color=RGB(tab10_primary_colour[1]-0.1,tab10_primary_colour[2]-0.1,tab10_primary_colour[3]-0.1), markersize=markersize, markerstrokewidth=0.0, alpha=1, labels=nothing,)

        end
    end

    scatter!([-1], [-1], color=RGB(tab10_primary_colour[1]-0.1,tab10_primary_colour[2]-0.1,tab10_primary_colour[3]-0.1), markersize=markersize, markerstrokewidth=0.0, alpha=1, label="Intersections",)

    #scatter!([-1,0,1], [0,0,0], 
    #            markersize = 6.0,
    #            markershape = :utriangle,
    #            markercolor = :black,#palette(:tab10)[8],
    #            markerstrokecolor = :black,
    #            markerstrokewidth = 0,#,0.5,
    #            alpha=1, labels=nothing,)
    #annotate!(-1, 0, text(" \$\\xi_1\$", :black, :bottom, 16))
    #annotate!(0, 0, text(" \$\\xi_2\$", :black, :bottom, 16))
    #annotate!(1, 0, text(" \$\\xi_3\$", :black, :bottom, 16))

    #title!("\$\\varepsilon=$ε\$, \$\\varrho=$ϱ\$")

    plot_ball_Qs = zeros(length(Qs))
    Threads.@threads for i in ProgressBar(eachindex(Qs))
        plot_ball_Qs[i] = in_W2_ball(P,ε,Qs[i])

    end

    tab10_primary_colour = [188, 189, 34]/255

    for i in ProgressBar(eachindex(Qs))
        if plot_ball_Qs[i] == 1
            mean, std = mean_and_std(Qs[i])
            scatter!([mean], [std], color=RGB(tab10_primary_colour[1],tab10_primary_colour[2],tab10_primary_colour[3]), markersize=markersize, markerstrokewidth=0.0, labels=nothing,)

        end
    end

    scatter!([-1], [-1], color=RGB(tab10_primary_colour[1],tab10_primary_colour[2],tab10_primary_colour[3]), markersize=markersize, markerstrokewidth=0.0, alpha=1, label="Concentration",)

    scatter!([-1,0,1], [0,0,0], 
                markersize = 6.0,
                markershape = :utriangle,
                markercolor = :black,#palette(:tab10)[8],
                markerstrokecolor = :black,
                markerstrokewidth = 0,#,0.5,
                alpha=1, labels=nothing,)
    annotate!(-1, 0, text(" \$\\xi_1\$", :black, :bottom, 16))
    annotate!(0, 0, text(" \$\\xi_2\$", :black, :bottom, 16))
    annotate!(1, 0, text(" \$\\xi_3\$", :black, :bottom, 16))

    display(plt)
    savefig(plt, "figures/ambiguity-sets.pdf")

end

