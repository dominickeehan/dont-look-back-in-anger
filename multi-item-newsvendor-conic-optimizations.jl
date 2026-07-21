using LinearAlgebra
using JuMP, MathOptInterface, Gurobi


# This file expects the following experiment constants to be defined in the
# main script before it is included:
#
# const number_of_items = 3
# const number_of_consumers = 1000


# Each Julia thread reuses its own single-threaded Gurobi environment.
const julia_thread_count =
    Threads.nthreads(:default) + Threads.nthreads(:interactive)
const gurobi_environments =
    Union{Nothing,Gurobi.Env}[nothing for _ in 1:julia_thread_count]
const gurobi_environment_locks =
    [ReentrantLock() for _ in 1:julia_thread_count]


function _gurobi_environment_for_current_thread()
    thread_id = Threads.threadid()
    environment = gurobi_environments[thread_id]
    if isnothing(environment)
        lock(gurobi_environment_locks[thread_id]) do
            if isnothing(gurobi_environments[thread_id])
                gurobi_environments[thread_id] = Gurobi.Env(
                    Dict{String,Any}("OutputFlag" => 0, "Threads" => 1),
                )
            end
            environment = gurobi_environments[thread_id]
        end
    end
    return environment::Gurobi.Env
end


const multi_item_optimizer = optimizer_with_attributes(
    () -> Gurobi.Optimizer(_gurobi_environment_for_current_thread()),
)
const multi_item_geometry_tolerance = 1.0e-6


function _new_multi_item_model()
    Problem = Model(multi_item_optimizer)
    set_string_names_on_creation(Problem, false)
    return Problem
end


function _optimize_multi_item_model!(Problem)
    optimize!(Problem)
    is_solved_and_feasible(Problem) && return nothing

    set_attribute(Problem, "BarHomogeneous", 1)
    set_attribute(Problem, "NumericFocus", 3)
    optimize!(Problem)
    is_solved_and_feasible(Problem) && return nothing

    error(
        "Gurobi did not solve the multi-item newsvendor model: " *
        "termination_status=$(termination_status(Problem)), " *
        "primal_status=$(primal_status(Problem))",
    )
end


function _normalized_positive_weights_and_demands(demands, weights)
    positive_weight_indices = weights .> 0.0
    weights = Float64.(weights[positive_weight_indices])
    weights = weights / sum(weights)
    demands = demands[positive_weight_indices]
    return demands, weights
end


# Construct the affine pieces of the newsvendor loss and the normalized box
# support [0,1]^number_of_items.
# See Corollary 2 of
# "Wasserstein Distributionally Robust Optimization with Heterogeneous Data
# Sources" by Rychener, Esteban-Perez, Morales, and Kuhn.
function _multi_item_newsvendor_problem_data(
    instance_underage_costs, instance_overage_costs,
)
    loss_pieces = collect(
        Iterators.product(fill([false, true], number_of_items)...),
    )
    a = [
        [
            loss_pieces[l][i] ? instance_underage_costs[i] :
            -instance_overage_costs[i]
            for i in 1:number_of_items
        ]
        for l in eachindex(loss_pieces)
    ]
    b = [-a[l] for l in eachindex(a)]
    C = [
        -Matrix{Float64}(I, number_of_items, number_of_items);
        Matrix{Float64}(I, number_of_items, number_of_items)
    ]
    g = [
        zeros(number_of_items);
        ones(number_of_items)
    ]
    return a, b, C, g
end


function SO_multi_item_newsvendor_objective_value_and_order(
    _,
    demands,
    weights,
    instance_underage_costs,
    instance_overage_costs,
)
    demands, weights =
        _normalized_positive_weights_and_demands(demands, weights)
    a, b, _, _ = _multi_item_newsvendor_problem_data(
        instance_underage_costs, instance_overage_costs,
    )
    T = length(demands)

    Problem = _new_multi_item_model()
    @variables(Problem, begin
        number_of_consumers >= order[i = 1:number_of_items] >= 0.0
        s[t = 1:T]
    end)

    for t in 1:T, l in eachindex(a)
        @constraint(
            Problem,
            sum(b[l][i] * order[i] for i in 1:number_of_items) +
            sum(a[l][i] * demands[t][i] for i in 1:number_of_items) <= s[t],
        )
    end

    @objective(Problem, Min, sum(weights[t] * s[t] for t in 1:T))
    _optimize_multi_item_model!(Problem)
    return objective_value(Problem), value.(order)
end


function _build_W2_DRO_multi_item_newsvendor_problem(
    demands,
    weights,
    instance_underage_costs,
    instance_overage_costs,
    normalized_epsilon,
)
    a, b, C, g = _multi_item_newsvendor_problem_data(
        instance_underage_costs, instance_overage_costs,
    )
    T = length(demands)

    Problem = _new_multi_item_model()
    @variables(Problem, begin
        1.0 >= order[i = 1:number_of_items] >= 0.0
        λ >= 0.0
        γ[t = 1:T]
        z[t = 1:T, l = 1:length(a), m = 1:length(g)] >= 0.0
        w[t = 1:T, l = 1:length(a), i = 1:number_of_items]
    end)

    for t in 1:T, l in eachindex(a)
        @constraint(
            Problem,
            [
                2.0 * λ;
                γ[t] -
                sum(b[l][i] * order[i] for i in 1:number_of_items) -
                sum(w[t, l, i] * demands[t][i] for i in 1:number_of_items) -
                sum(z[t, l, m] * g[m] for m in eachindex(g));
                [w[t, l, i] for i in 1:number_of_items]
            ] in MathOptInterface.RotatedSecondOrderCone(number_of_items + 2),
        )
        for i in 1:number_of_items
            @constraint(
                Problem,
                a[l][i] -
                sum(C[m, i] * z[t, l, m] for m in eachindex(g)) ==
                w[t, l, i],
            )
        end
    end

    @objective(
        Problem,
        Min,
        normalized_epsilon^2 * λ +
        sum(weights[t] * γ[t] for t in 1:T),
    )
    return Problem, order
end


function _solve_W2_DRO_multi_item_newsvendor_problem!(
    Problem, order,
)
    _optimize_multi_item_model!(Problem)
    return (
        number_of_consumers * objective_value(Problem),
        number_of_consumers .* value.(order),
    )
end


function W2_DRO_multi_item_newsvendor_objective_value_and_order(
    ε,
    demands,
    weights,
    instance_underage_costs,
    instance_overage_costs,
)
    if ε == 0.0
        return SO_multi_item_newsvendor_objective_value_and_order(
            ε,
            demands,
            weights,
            instance_underage_costs,
            instance_overage_costs,
        )
    end

    demands, weights =
        _normalized_positive_weights_and_demands(demands, weights)
    normalized_demands = [demand ./ number_of_consumers for demand in demands]
    Problem, order =
        _build_W2_DRO_multi_item_newsvendor_problem(
            normalized_demands,
            weights,
            instance_underage_costs,
            instance_overage_costs,
            ε / number_of_consumers,
        )
    return _solve_W2_DRO_multi_item_newsvendor_problem!(
        Problem, order,
    )
end


include("multi-item-newsvendor-intersection-conic-optimizations.jl")
