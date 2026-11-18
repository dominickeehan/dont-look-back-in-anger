using Plots, Measures

# Plots the weights for the weight parameter grids of the experiments to check their ranges.

include("weights.jl")

default() # Reset plot settings to defaults.

# A 3:2 aspect ratio for the plot, excluding + 6 padding of the top, bottom, and left margins.
# The sqrt(3) conversion means that (275 + 6 + 6, 183 + 6) is the size of the plot in points to
# specify in latex to correctly size the embedded fonts.
# (275, 183) itself is just visually nice.
gr(size = (275 + 6 + 6, 183 + 6) .* sqrt(3))
fontfamily = "Computer Modern" # Close to Latin Modern.
default(framestyle = :box,
        grid = true,
        gridlinewidth = 0.5,
        gridalpha = 0.075,
        #minorgrid = true, # No minor grid for unscaled axes.
        #minorgridlinewidth = 1.0,
        #minorgridalpha = 0.075,
        #minorgridlinestyle = :dash,
        tick_direction = :in,
        xminorticks = 0,
        yminorticks = 0,
        fontfamily = fontfamily,
        titlefont = Plots.font(fontfamily, pointsize = 12), # Slight over emphasis on axis labels.
        guidefont = Plots.font(fontfamily, pointsize = 12), # Slight over emphasis on axis labels.
        legendfont = Plots.font(fontfamily, pointsize = 11),
        tickfont = Plots.font(fontfamily, pointsize = 10)) # Slight under emphasis on tick labels.


LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))

T = 100

# Windowing weights.
plt = plot()
for window_size in unique(floor.(Int, LogRange(1,100,30))); plot!(1:T, windowing_weights(T, window_size), label = nothing, color = :black); end
display(plt)

# Smoothing weights.
plt = plot()
for α in [[0]; LogRange(1e-4,1e0,30)]; plot!(1:T, smoothing_weights(T, α), label = nothing, color = :black); end
display(plt)

# 1-Wasserstein weights with a linear drift profile.
plt = plot()
for ρ╱ε in [[0]; LogRange(1e-4,1e0,30)]; plot!(1:T, Wp_power_law_drift_profile_weights(1, T, ρ╱ε, 1), label = nothing, color = :black); end
display(plt)

# 2-Wasserstein weights with a linear drift profile.
plt = plot()
for ρ╱ε in [[0]; LogRange(1e-4,1e0,30)]; plot!(1:T, Wp_power_law_drift_profile_weights(2, T, ρ╱ε, 1), label = nothing, color = :black); end
display(plt)

# 1-Wasserstein weights for ρ╱ε = 0 and the smallest positive ρ╱ε.
plt = plot(1:T, Wp_power_law_drift_profile_weights(1, T, 0, 1), label = nothing, color = :black)
plot!(1:T, Wp_power_law_drift_profile_weights(1, T, 1e-4, 1), label = nothing, color = :black)
display(plt)

# 2-Wasserstein weights for ρ╱ε = 0 and the smallest positive ρ╱ε.
plt = plot(1:T, Wp_power_law_drift_profile_weights(2, T, 0, 1), label = nothing, color = :black)
plot!(1:T, Wp_power_law_drift_profile_weights(2, T, 1e-4, 1), label = nothing, color = :black)
display(plt)

