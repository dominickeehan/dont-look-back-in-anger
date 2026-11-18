using Random, Distributions
using ProgressBars

# Tests the dual solvers of multi-item-newsvendor-dual-optimizations.jl against the conic programs
# of multi-item-newsvendor-conic-optimizations.jl on random problems.

# Each file goes in its own module as both define the same functions.
module Dual
        number_of_items = 3
        number_of_consumers = 1000

        include("weights.jl")
        include("multi-item-newsvendor-dual-optimizations.jl")
end

module Conic
        number_of_items = 3
        number_of_consumers = 1000

        include("weights.jl")
        include("multi-item-newsvendor-conic-optimizations.jl")
end

Random.seed!(42)

# Loop through the problems and compare the two solvers on 200 random instances of each.
for problem in ["SO", "W2_DRO", "REMK_intersection_W2_DRO"]

        dual_objective_value_and_order = getfield(Dual, Symbol(problem*"_multi_item_newsvendor_objective_value_and_order"))
        conic_objective_value_and_order = getfield(Conic, Symbol(problem*"_multi_item_newsvendor_objective_value_and_order"))

        objective_value_differences = Float64[]
        order_differences = Float64[]

        for _ in ProgressBar(1:200)

                # Random instance with three items.
                T = rand(10:50)
                demands = [1000*rand(3) for _ in 1:T]
                underage_costs = rand(3.0:9.0, 3)
                overage_costs = ones(3)

                ε = rand(LogUniform(10, 2000))
                weights = problem == "REMK_intersection_W2_DRO" ? Dual.REMK_intersection_weights(T, rand(LogUniform(1e-3, 1))) : Dual.smoothing_weights(T, rand(LogUniform(1e-2, 1)))

                dual_objective_value, dual_order = dual_objective_value_and_order(ε, demands, weights, underage_costs, overage_costs)
                conic_objective_value, conic_order = conic_objective_value_and_order(ε, demands, weights, underage_costs, overage_costs)

                push!(objective_value_differences, abs(dual_objective_value - conic_objective_value)/max(conic_objective_value, 1))
                push!(order_differences, maximum(abs.(dual_order - conic_order)))

        end

        # Print the largest differences. Orders differ most for non-intersecting balls in the intersection
        # approach where Gurobi's shared point needs a slightly larger radius increase.
        println(problem)
        println("Largest relative difference in objective value: ", maximum(objective_value_differences))
        println("Largest difference in order (demands are up to 1000): ", maximum(order_differences))

end
