using Random, Distributions, Statistics, StatsBase
using JuMP, Gurobi
using Plots, Measures
using ProgressBars

# The first part of this script takes hard-coded data giving the means and standard deviations of uniform distributions within W2 balls. 
# For code which samples the distributions from these balls, see line 195.

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

plt = plot(xlims=(-4,8),
            ylims=(0,7),
            xlabel="Mean", 
            ylabel="Standard deviation",
            #legend=:horizontal,
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 6pt, 
            leftmargin = 6pt)

function nonnegative_ellipse_coords(ellipse_parameters)

    horizontal_radius, vertical_radius, x_centre, y_centre = ellipse_parameters

    t = range(1.5*π, -0.5*π; length=1000)

    x = x_centre .+ horizontal_radius .* cos.(t)
    y = y_centre .+ vertical_radius .* sin.(t)
    
    negative_y_indices = y .< 0
    y[negative_y_indices] .= 0

    return x, y

end

# Saved data from /figures/ambiguity-sets-data-*.pdf
intersection_ellipse_1 = [5.9,5.9,1,0] # Width, height, x, y. 
intersection_ellipse_2 = [5.4,5.4,-1,0]
intersection_ellipse_3 = [5.1,5.1,2,0]
intersection_ellipse_4 = [4.9,4.9,3,0]
weighted_ellipse = [2.75,2.75,1.75,1.5]

linewidth = 1.5
alpha = 1
fillalpha = 0.075

x_coords, y_coords = nonnegative_ellipse_coords(intersection_ellipse_1)
plot!(x_coords,
        y_coords,
        color = palette(:tab10)[1],
        linewidth = 0,
        linestyle = :solid,
        alpha = 0,
        label = nothing,
        fill = (0, fillalpha, palette(:tab10)[1]))

x_coords, y_coords = nonnegative_ellipse_coords(intersection_ellipse_2)
plot!(x_coords,
        y_coords,
        color = palette(:tab10)[1],
        linewidth = 0,
        linestyle = :solid,
        alpha = 0,
        label = nothing,
        fill = (0, fillalpha, palette(:tab10)[1]))

x_coords, y_coords = nonnegative_ellipse_coords(intersection_ellipse_3)
plot!(x_coords,
        y_coords,
        color = palette(:tab10)[1],
        linewidth = 0,
        linestyle = :solid,
        alpha = 0,
        label = nothing,
        fill = (0, fillalpha, palette(:tab10)[1]))

x_coords, y_coords = nonnegative_ellipse_coords(intersection_ellipse_4)
plot!(x_coords,
        y_coords,
        color = palette(:tab10)[1],
        linewidth = 0,
        linestyle = :solid,
        alpha = 0,
        label = nothing,
        fill = (0, fillalpha, palette(:tab10)[1]))


function nonnegative_intersected_ellipse_coords(ellipse_1_parameters, ellipse_2_parameters)

    horizontal_radius_2, vertical_radius_2, x_centre_2, y_centre_2 = ellipse_2_parameters

    x_coords, y_coords = nonnegative_ellipse_coords(ellipse_1_parameters) 

    for i in eachindex(x_coords)
        if i ∈ [1,250]∪[751,1000] # Negative quadrant so take negative root.
            try
                y_coords[i] = min(y_coords[i], y_centre_2 - vertical_radius_2*sqrt(1-((x_coords[i]-x_centre_2)/horizontal_radius_2)^2))

            catch # If outside shared x domain.
                y_coords[i] = -99

            end

        else # Positive quadrant so take positive root.
            try
                y_coords[i] = min(y_coords[i], y_centre_2 + vertical_radius_2*sqrt(1-((x_coords[i]-x_centre_2)/horizontal_radius_2)^2))

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

    return [x_coords[1]; x_coords], [0; y_coords] # Prepend to cover intial gap.

end


x_coords, y_coords = nonnegative_intersected_ellipse_coords(intersection_ellipse_2, intersection_ellipse_4)
plot!(x_coords,
        y_coords,
        color = palette(:tab10)[1],
        linewidth = linewidth,
        linestyle = :solid,
        alpha = 1,
        label = nothing)

x_coords, y_coords = nonnegative_ellipse_coords([1,1,-1,-1])
plot!(x_coords,
        y_coords,
        color = palette(:tab10)[1],
        linewidth = linewidth,
        linestyle = :solid,
        alpha = alpha,
        label = "Intersection",
        fill = (0, 0.268, palette(:tab10)[1]))

x_coords, y_coords = nonnegative_ellipse_coords(weighted_ellipse)
    plot!(x_coords,
        y_coords,
        color = palette(:tab10)[2],
        linewidth = linewidth,
        linestyle = :dash,
        alpha = alpha,
        label = "Weighted",
        fill = (0, 0.268, palette(:tab10)[2]))

samples = [1,-1,2,3]
scatter!(samples, zeros(length(samples)), 
            markersize = 6.0,
            markershape = :utriangle,
            markercolor = :black,
            markerstrokecolor = :black,
            markerstrokewidth = 0,
            alpha = 1,
            labels = nothing)

for i in eachindex(samples)
    annotate!(samples[i], 0, text(" \$\\xi_$i\$", :black, :bottom, 12))
end

display(plt)
savefig(plt, "figures/ambiguity-sets.pdf")


if false # Sample from the intersection-based and weight-based ambiguity sets.

    p = 2

    env = Gurobi.Env()
    GRBsetintparam(env, "OutputFlag", 0)
    Linear_Optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

    function in_ball(P, ε, Q)

        # P, Q := [points, weights] .

        m = length(P[1])
        n = length(Q[1])

        d = [abs(ξ - ζ)^p for ξ in P[1], ζ in Q[1]]

        model = Model(Linear_Optimizer)

        @variable(model, γ[1:m, 1:n] >= 0)

        @constraint(model, [i=1:m], sum(γ[i,j] for j in 1:n) == P[2][i])
        @constraint(model, [j=1:n], sum(γ[i,j] for i in 1:m) == Q[2][j])

        @objective(model, Min, sum(γ[i,j] * d[i,j] for i in 1:m, j in 1:n))

        optimize!(model)

        @assert is_solved_and_feasible(model)

        try; return ifelse((objective_value(model)^(1/p)) <= ε, 1, 0); catch; return 0; end
    end

    include("weights.jl")

    function mean_and_std(Q)

        return mean(Q[1], Weights(Q[2])), std(Q[1], Weights(Q[2]))
    end

    Ξ = [-4,8]
    max_σ = 7
    samples = [1,-1,2,3]

    number_of_points = 100
    number_of_distributions = 10000 # 30000

    markersize = 4

    weighted_ε = 3
    weighted_ρ = 1/3

    scale_radii = 1.5
    intersection_ε = scale_radii*weighted_ε
    intersection_ρ = weighted_ρ

    P = [samples, Wp_weights(p, length(samples), weighted_ρ/weighted_ε)]

    display(P)

    Qs = [[zeros(number_of_points), zeros(number_of_points)] for _ in 1:number_of_distributions]
    Threads.@threads for i in ProgressBar(eachindex(Qs))
        local n = number_of_points
        local μ = rand(Uniform(Ξ[1],Ξ[end]))
        local σ = rand(Uniform(0,max_σ))
        #local points = rand(Normal(μ, σ), n)
        local points = rand(Uniform(μ-sqrt(3)*σ, μ+sqrt(3)*σ), n)

        Qs[i] = [points, 1/n*ones(n)]

    end

    plot_intersection_Qs = [zeros(length(samples)) for _ in eachindex(Qs)]
    Threads.@threads for i in ProgressBar(eachindex(Qs))
        plot_intersection_Qs[i] = [in_ball([[P[1][j]],[1.0]],intersection_ε+(length(samples)-j+1)*intersection_ρ,Qs[i]) for j in 1:length(samples)]

    end

    default() # Reset plot defaults.

    gr(size = (275+6,183+6).*sqrt(3))

    font_family = "Computer Modern"
    primary_font = Plots.font(font_family, pointsize = 12)
    secondary_font = Plots.font(font_family, pointsize = 10)
    legend_font = Plots.font(font_family, pointsize = 11)

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

    plt = plot(xlims=(Ξ[1],Ξ[end]), 
                ylims=(0,max_σ), 
                xlabel="Mean", 
                ylabel="Standard deviation",           
                topmargin = 0pt, 
                rightmargin = 0pt,
                bottommargin = 6pt, 
                leftmargin = 6pt)

    color_increment = 0.05
    pink = min.([227,119,194]/255 .+ length(samples)*color_increment, 1)

    for i in eachindex(samples)
        for j in ProgressBar(eachindex(Qs))

            if sum(plot_intersection_Qs[j]) == i
                mean, std = mean_and_std(Qs[j])
                intersection_color = max.(pink .- i*color_increment,0)
                scatter!([mean], [std], 
                            color = RGB(intersection_color[1],intersection_color[2],intersection_color[3]), 
                            markersize = markersize,
                            markerstrokewidth = 0.0,
                            alpha = 1,
                            labels = nothing)

            end
        end
    end

    intersection_color = pink .- length(samples)*color_increment
    scatter!([-1], [-1], 
                color = RGB(intersection_color[1],intersection_color[2],intersection_color[3]),
                markersize = markersize, 
                markerstrokewidth = 0.0, 
                alpha = 1, 
                label = "Intersection")

    scatter!(samples, zeros(length(samples)), 
                markersize = 6.0,
                markershape = :utriangle,
                markercolor = :black,
                markerstrokecolor = :black,
                markerstrokewidth = 0,
                alpha = 1,
                labels = nothing)

    for i in eachindex(samples)
        annotate!(samples[i], 0, text(" \$\\xi_$i\$", :black, :bottom, 12))
    end

    display(plt)
    savefig(plt, "figures/intersection-ambiguity-set-data.pdf")

    default() # Reset plot defaults.

    gr(size = (275+6,183+6).*sqrt(3))

    font_family = "Computer Modern"
    primary_font = Plots.font(font_family, pointsize = 12)
    secondary_font = Plots.font(font_family, pointsize = 10)
    legend_font = Plots.font(font_family, pointsize = 11)

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

    plt = plot(xlims=(Ξ[1],Ξ[end]), 
                ylims=(0,max_σ), 
                xlabel="Mean", 
                ylabel="Standard deviation",
                topmargin = 0pt, 
                rightmargin = 0pt,
                bottommargin = 6pt, 
                leftmargin = 6pt)

    plot_ball_Qs = zeros(length(Qs))
    Threads.@threads for i in ProgressBar(eachindex(Qs))
        plot_ball_Qs[i] = in_ball(P,weighted_ε,Qs[i])

    end

    olive = [188, 189, 34]/255

    for i in ProgressBar(eachindex(Qs))
        if plot_ball_Qs[i] == 1
            mean, std = mean_and_std(Qs[i])
            scatter!([mean], [std], 
                        color = RGB(olive[1],olive[2],olive[3]), 
                        markersize = markersize, 
                        markerstrokewidth = 0.0, 
                        labels = nothing)

        end
    end

    scatter!([-1], [-1], 
                color = RGB(olive[1],olive[2],olive[3]), 
                markersize = markersize, 
                markerstrokewidth = 0.0, 
                alpha = 1, 
                label = "Weighted")

    scatter!(samples, zeros(length(samples)), 
                markersize = 6.0,
                markershape = :utriangle,
                markercolor = :black,
                markerstrokecolor = :black,
                markerstrokewidth = 0,
                alpha = 1,
                labels = nothing)

    for i in eachindex(samples)
        annotate!(samples[i], 0, text(" \$\\xi_$i\$", :black, :bottom, 12))

    end

    display(plt)
    savefig(plt, "figures/weighted-ambiguity-set-data.pdf")

end