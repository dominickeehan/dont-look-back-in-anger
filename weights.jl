# Dominic Keehan : 2025

function SES_weights(history_of_observations, α)

    T = length(history_of_observations)

    weights = [α*(1-α)^(t-1) for t in T:-1:1]
    weights .= weights/sum(weights)

    return weights
end

function bowing_weights(history_of_observations, parameters)

    #T = length(history_of_observations)

    T = parameters[1]
    α = parameters[2]

    weights = [α*(1-α)^(t-1) for t in T:-1:1]
    weights .= weights/sum(weights)

    weights = [max(1-weights[t],0) for t in T:-1:1]
    weights .= weights/sum(weights)

    #return_weights = zeros(length(history_of_observations))
    #return_weights[end-(T-1):end] = weights

    return vcat(zeros(length(history_of_observations)-T), weights)
end

function windowing_weights(history_of_observations, window_size)

    T = length(history_of_observations)

    weights = zeros(T)

    if window_size >= T
        weights .= 1
    else
        for t in T:-1:T-(window_size-1)
            weights[t] = 1
        end
    end

    weights .= weights/sum(weights)

    return weights
end