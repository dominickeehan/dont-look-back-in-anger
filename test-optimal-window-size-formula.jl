using Random, Distributions

Random.seed!(42)

true_objective_value(t, ε, ρ) = t * ((max(ε - (0.5)*(t+1)*ρ, 0)^2.0))

true_optimal_window_size(T, ε, ρ) = argmax([true_objective_value(t, ε, ρ) for t in 1:T])

function formula_for_optimal_window_size(T, ε, ρ)
    unprojected_formula = (1.0/(3.0)) * (2.0*(ε/ρ) - 1.0)
    
    projected_floor_formula = max(min(floor(unprojected_formula), T), 1)
    objective_projected_floor_formula = true_objective_value(projected_floor_formula, ε, ρ)

    projected_ceil_formula = max(min(ceil(unprojected_formula), T), 1)
    objective_projected_ceil_formula = true_objective_value(projected_ceil_formula, ε, ρ)

    if objective_projected_floor_formula >= objective_projected_ceil_formula

        return projected_floor_formula
    else

        return projected_ceil_formula
    end
end

using ProgressBars
for _ in ProgressBar(1:1000)
    T = rand(1:100)
    ε = rand(Uniform(0,1))
    ρ = rand(Uniform(0,1))

    optimal_window_size = true_optimal_window_size(T, ε, ρ)
    formula_window_size = formula_for_optimal_window_size(T, ε, ρ)

    if optimal_window_size != formula_window_size
        display([optimal_window_size, formula_window_size, T, ε, ρ])

    end
end
