using Random, Distributions, Statistics, StatsBase
using JuMP, Gurobi
using Plots
using ProgressBars

Random.seed!(42)

env = Gurobi.Env()
GRBsetintparam(env, "OutputFlag", 0)
Linear_Optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

function in_W2_ball(P, ε, Q)
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
number_of_distributions = 1000
base_markersize = 8
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

    # Calculate all means and stds for density detection
    all_means = [mean_and_std(Qs[i])[1] for i in eachindex(Qs)]
    all_stds = [mean_and_std(Qs[i])[2] for i in eachindex(Qs)]

    # Density detection parameters
    dense_threshold = 0.3
    dense_markersize = base_markersize * 2.5
    sparse_markersize = base_markersize
    dense_sample_ratio = 0.1

    # Classify points
    is_dense = zeros(Bool, length(Qs))
    for i in eachindex(Qs)
        mean_val, std_val = mean_and_std(Qs[i])
        nearby = 0
        for j in eachindex(Qs)
            if i != j && sqrt((all_means[j] - mean_val)^2 + (all_stds[j] - std_val)^2) < dense_threshold
                nearby += 1
                if nearby > 5  # If at least 5 nearby points, consider dense
                    is_dense[i] = true
                    break
                end
            end
        end
    end

    # Set up plot
    gr(size = (600,400))
    font_family = "Computer Modern"
    primary_font = Plots.font(font_family, pointsize = 17)
    secondary_font = Plots.font(font_family, pointsize = 11)
    legend_font = Plots.font(font_family, pointsize = 15)

    default(framestyle = :box,
            grid = true,
            gridalpha = 0.075,
            tick_direction = :in,
            xminorticks = 0, 
            yminorticks = 0,
            fontfamily = font_family,
            guidefont = primary_font,
            tickfont = secondary_font,
            legendfont = legend_font)

    plt = plot(xlims=(support[1],support[end]), ylims=(0,3), xlabel="Mean", ylabel="Standard deviation")

    tab10_primary_colour = [227,119,194]/256

    # Plot dense points (reduced count, larger size)
    for i in ProgressBar(eachindex(Qs))
        if is_dense[i] && rand() < dense_sample_ratio
            mean, std = mean_and_std(Qs[i])
            if sum(plot_intersection_Qs[i]) == 1
                scatter!([mean], [std], color=RGB(min(tab10_primary_colour[1]+0.1,1.0),tab10_primary_colour[2]+0.1,tab10_primary_colour[3]+0.1), 
                        markersize=dense_markersize, markerstrokewidth=0.0, alpha=1, labels=nothing)
            elseif sum(plot_intersection_Qs[i]) == 2
                scatter!([mean], [std], color=RGB(tab10_primary_colour[1],tab10_primary_colour[2],tab10_primary_colour[3]), 
                        markersize=dense_markersize, markerstrokewidth=0.0, alpha=1, labels=nothing)
            elseif sum(plot_intersection_Qs[i]) == 3
                scatter!([mean], [std], color=RGB(tab10_primary_colour[1]-0.1,tab10_primary_colour[2]-0.1,tab10_primary_colour[3]-0.1), 
                        markersize=dense_markersize, markerstrokewidth=0.0, alpha=1, labels=nothing)
            end
        end
    end

    # Plot sparse points (all perimeter points)
    for i in ProgressBar(eachindex(Qs))
        if !is_dense[i]
            mean, std = mean_and_std(Qs[i])
            if sum(plot_intersection_Qs[i]) == 1
                scatter!([mean], [std], color=RGB(min(tab10_primary_colour[1]+0.1,1.0),tab10_primary_colour[2]+0.1,tab10_primary_colour[3]+0.1), 
                        markersize=sparse_markersize, markerstrokewidth=0.0, alpha=1, labels=nothing)
            elseif sum(plot_intersection_Qs[i]) == 2
                scatter!([mean], [std], color=RGB(tab10_primary_colour[1],tab10_primary_colour[2],tab10_primary_colour[3]), 
                        markersize=sparse_markersize, markerstrokewidth=0.0, alpha=1, labels=nothing)
            elseif sum(plot_intersection_Qs[i]) == 3
                scatter!([mean], [std], color=RGB(tab10_primary_colour[1]-0.1,tab10_primary_colour[2]-0.1,tab10_primary_colour[3]-0.1), 
                        markersize=sparse_markersize, markerstrokewidth=0.0, alpha=1, labels=nothing)
            end
        end
    end

    # Add legend items
    scatter!([-1], [-1], color=RGB(tab10_primary_colour[1]-0.1,tab10_primary_colour[2]-0.1,tab10_primary_colour[3]-0.1), 
            markersize=sparse_markersize, markerstrokewidth=0.0, alpha=1, label="Intersections")

    # Plot W2 ball points
    plot_ball_Qs = zeros(length(Qs))
    Threads.@threads for i in ProgressBar(eachindex(Qs))
        plot_ball_Qs[i] = in_W2_ball(P,ε,Qs[i])
    end

    tab10_primary_colour = [188, 189, 34]/256

    for i in ProgressBar(eachindex(Qs))
        if plot_ball_Qs[i] == 1
            mean, std = mean_and_std(Qs[i])
            current_size = is_dense[i] ? dense_markersize : sparse_markersize
            scatter!([mean], [std], color=RGB(tab10_primary_colour[1],tab10_primary_colour[2],tab10_primary_colour[3]), 
                    markersize=current_size, markerstrokewidth=0.0, labels=nothing)
        end
    end

    scatter!([-1], [-1], color=RGB(tab10_primary_colour[1],tab10_primary_colour[2],tab10_primary_colour[3]), 
            markersize=sparse_markersize, markerstrokewidth=0.0, alpha=1, label="Concentration")

    # Add triangle markers
    scatter!([-1,0,1], [0,0,0], 
            markersize = 6.0,
            markershape = :utriangle,
            markercolor = :black,
            markerstrokecolor = :black,
            markerstrokewidth = 0,
            alpha=1, labels=nothing)
    annotate!(-1, 0, text(" \$\\xi_1\$", :black, :bottom, 16))
    annotate!(0, 0, text(" \$\\xi_2\$", :black, :bottom, 16))
    annotate!(1, 0, text(" \$\\xi_3\$", :black, :bottom, 16))

    display(plt)
    #savefig(plt, "figures/ambiguity-sets.pdf")
end