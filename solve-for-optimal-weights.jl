using JuMP, Ipopt
using Plots

T = 60
p = 2

function solve_for_weights(ϵ, ρ)

    Problem = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "tol" => 1e-9))

    @variable(Problem, 1>= w[t=1:T] >=0)

    @constraint(Problem, sum(w[t] for t in 1:T) == 1)
    @constraint(Problem, (sum(w[t]*t^p for t in 1:T)*ρ^p)^(1/p) <= (ϵ)^(1/p))
    for t in 1:T-1; @constraint(Problem, w[t] >= w[t+1]); end

    @objective(Problem, Max, (1/(sum(w[t]^2 for t in 1:T)))*(((ϵ)^(1/p)-(sum(w[t]*t^p for t in 1:T)*ρ^p)^(1/p))^(2*p)))

    optimize!(Problem)

    display(objective_value(Problem))

    weights = [value(w[t]) for t in 1:T]

    display(plot(1:T, weights))

    return reverse(weights)

end

weights = solve_for_weights(10.0,0.05)