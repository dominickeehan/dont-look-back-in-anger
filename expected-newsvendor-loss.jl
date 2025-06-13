using Distributions

D = 10000
demand_probability = 0.001
Cu = 5.0
Co = 2.0
order = 100.1

function expected_newsvendor_loss_analytical()

    a = cdf(Binomial(D-1,demand_probability), order-1)
    b = cdf(Binomial(D,demand_probability), order)

    expected_underage_cost = Cu * (D*demand_probability*(1-a) - order*(1-b))
    expected_overage_cost = Co * (order*b - D*demand_probability*a)

    return expected_underage_cost + expected_overage_cost
end

function expected_newsvendor_loss_numerical()

    demand_distribution = Binomial(D, demand_probability)
    return sum((Cu*max(demand-order,0) + Co*max(order-demand,0)) * pdf(demand_distribution, demand) for demand in 0:D)

end

analytical_result = expected_newsvendor_loss_analytical()
numerical_result = expected_newsvendor_loss_numerical()

println("Analytical result: ", analytical_result)
println("Numerical result: ", numerical_result)
println("Difference: ", abs(analytical_result - numerical_result))

