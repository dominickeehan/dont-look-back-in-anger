const simulation_seed = 42

const underage_cost_stream = 1
const overage_cost_stream = 2
const mode_weight_stream = 3
const demand_stream = 4
const initial_probability_stream = 5
const mode_stream = 6
const innovation_stream = 7


function experiment_rng(
    global_repetition_index,
    stream;
    dimension = 0,
)
    # Keep each repetition, random quantity, and item on a deterministic stream
    # that does not depend on surrounding loop order. The dimension argument is
    # an item index, never a number of items, so an instance with more items
    # reuses every stream of the instances with fewer items.
    seed =
        simulation_seed +
        1_000_000 * global_repetition_index +
        1_000 * dimension +
        stream
    return MersenneTwister(seed)
end
