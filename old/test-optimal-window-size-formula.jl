using Random, Distributions

Random.seed!(42)

objective_value(t, ϵ, ρ, p) = t * ((max(ϵ - (0.5)*(t+1)*ρ, 0)^p))

true_optimal_window_size(T, ϵ, ρ, p) = argmax([objective_value(t, ϵ, ρ, p) for t in 1:T])

function formula_for_optimal_window_size(T, ϵ, ρ, p)

    unprojected_formula = (1.0/(p+1.0)) * (2.0*(ϵ/ρ) - 1.0)
    
    projected_floor_formula = max(min(floor(unprojected_formula), T), 1)
    objective_projected_floor_formula = objective_value(projected_floor_formula, ϵ, ρ, p)

    projected_ceil_formula = max(min(ceil(unprojected_formula), T), 1)
    objective_projected_ceil_formula = objective_value(projected_ceil_formula, ϵ, ρ, p)

    if objective_projected_floor_formula >= objective_projected_ceil_formula
        return projected_floor_formula
    else
        return projected_ceil_formula
    end
end


using ProgressBars
for _ in ProgressBar(1:1000000)

    T = rand(1:100)
    ϵ = rand(Uniform(0,1))
    ρ = rand(Uniform(0,1))
    p = rand(Uniform(0,3))

    optimal_window_size = true_optimal_window_size(T, ϵ, ρ, p)
    formula_window_size = formula_for_optimal_window_size(T, ϵ, ρ, p)

    if optimal_window_size != formula_window_size
        display([optimal_window_size, formula_window_size, T, ϵ, ρ, p])
    end
end