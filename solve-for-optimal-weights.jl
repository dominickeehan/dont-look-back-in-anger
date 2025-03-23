using JuMP, Ipopt
using Plots

T = 100

function solve_for_weights(ϵ, ρ)

    Problem = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "tol" => 1e-36))

    @variables(Problem, begin
                            1>= w[t=1:T] >=0 
                        end)

    @constraint(Problem, sum(w[t] for t in 1:T) == 1)
    for t in 1:T-1
        @constraint(Problem, w[t] >= w[t+1])
    end

    @objective(Problem, Max, (1/(sum(w[t]^2 for t in 1:T)))*(max(ϵ-sum(w[t]*t for t in 1:T)*ρ,0))^2)

    optimize!(Problem)

    weights = [value(w[t]) for t in 1:T]

    display(plot(1:T, weights))
    println(weights)

end

solve_for_weights(0.1,0.001)