# Intersection approach as conic programs. The ambiguity set is the intersection of the
# 2-Wasserstein balls of radii REMK_intersection_ball_radii around the K demands. If the balls do
# not intersect then their radii are all increased by the smallest amount that makes them intersect.


# Smallest amount added to every radius for the balls around the centres to share a point (negative
# if they already overlap). Returns this amount and the shared point.
function smallest_radius_increase(centres, radii)

        Problem = Model(optimizer)

        @variables(Problem, begin
                                1.0 >= ξ[i=1:number_of_items] >= 0.0
                                radius_increase
                            end)

        for k in eachindex(centres)
                @constraint(Problem, [radii[k] + radius_increase; ξ - centres[k]] in MathOptInterface.SecondOrderCone(number_of_items + 1))

        end

        @objective(Problem, Min, radius_increase)

        optimize_with_high_precision_fallback!(Problem)

        return value(radius_increase), value.(ξ)
end


# 2-Wasserstein DRO over the intersection of the balls around the demands.
function REMK_intersection_W2_DRO_multi_item_newsvendor_objective_value_and_order(ε, demands, weights, underage_costs, overage_costs)

        a, b, C, g = problem_matrices(underage_costs, overage_costs)

        # Demands and radii divided by the number of consumers to keep the problem numerically stable
        # (undone on return).
        K = length(demands)
        d = [demand/number_of_consumers for demand in demands]
        radii = REMK_intersection_ball_radii(K, ε, weights[end])/number_of_consumers # Every entry of weights is ρ╱ε (see REMK_intersection_weights).

        # Smallest increase in the radii for which the balls share a point. A distribution within r of the
        # point mass at d[k] has ‖ mean - d[k] ‖^2 + E‖ demand - mean ‖^2 <= r^2. So if the balls only meet
        # after the increase then the only distribution left is the point mass at the shared point ξ. In
        # this case, ordering ξ costs nothing as the demand is ξ for certain.
        increase, ξ = smallest_radius_increase(d, radii)
        if increase >= -1e-6; return 0.0, number_of_consumers*ξ; end

        # Conic program from Corollary 2 of Rychener et al. with a point mass at each demand.
        Problem = Model(optimizer)

        @variables(Problem, begin
                                1.0 >= order[i=1:number_of_items] >= 0.0
                                λ[k=1:K] >= 0.0
                                η
                                z[l=1:length(a),m=1:length(g)] >= 0.0
                                w[l=1:length(a),k=1:K,i=1:number_of_items]
                                s[l=1:length(a),k=1:K] >= 0.0
                            end)

        for l in eachindex(a)

                @constraints(Problem, begin
                                        # b[l]'*order + sum(w[l,k,:]'*d[k] + (1/4)*(1/λ[k])*w[l,k,:]'*w[l,k,:] for k in 1:K) + z[l,:]'*g <= η
                                        # <==>
                                        b[l]'*order + sum(w[l,k,:]'*d[k] + s[l,k] for k in 1:K) + z[l,:]'*g <= η

                                        a[l] - C'*z[l,:] .== sum(w[l,k,:] for k in 1:K)
                                      end)

                for k in 1:K
                        # w[l,k,:]'*w[l,k,:] <= 2*(2*λ[k])*s[l,k]
                        @constraint(Problem, [2.0*λ[k]; s[l,k]; w[l,k,:]] in MathOptInterface.RotatedSecondOrderCone(number_of_items + 2))

                end
        end

        @objective(Problem, Min, sum((radii[k]^2)*λ[k] for k in 1:K) + η)

        optimize_with_high_precision_fallback!(Problem)

        return number_of_consumers*objective_value(Problem), number_of_consumers*value.(order)
end
