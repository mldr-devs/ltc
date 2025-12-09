import SymbolicRegression: Options, equation_search, HallOfFame, SearchState
using Random, DataStructures


"""Noise function for SR"""
r(x) = randn(eltype(x), size(x)...)

function default_options()
    Options(
        binary_operators=[+, *, /, -],
        unary_operators=[cos, exp, sin, r],
        populations=20,
        save_to_file=false
    )
end

struct SymQ
    state::Union{Nothing, SearchState}
    hof::Union{Nothing, HallOfFame}
end



mutable struct SRAgent
    model::Union{Nothing, SymQ}
    target_model::Union{Nothing, SymQ}
    options::Options
    replay_buffer::CircularBuffer{SARSD}
    step::Int64

    function SRAgent(options::Options, replay_buffer_capacity::Int)
        new(nothing, nothing, options, CircularBuffer{SARSD}(replay_buffer_capacity), 0)
    end
end

function train!(agent::SRAgent)
    X = randn(2, 100)
    y = 2 * cos.(X[2, :]) + X[1, :] .^ 2 .- 2

    fit_result = equation_search(X, y; options=agent.options, n_epochs=1, return_state=true, saved_state=agent.model.state)
    agent.model = SymQ(fit_result.state, fit_result.hof)

    
    agent.step += 1

    if agent.step % 10 == 0
        agent.target_model = deepcopy(agent.model)
    end
end

function take_action(agent::SRAgent, observations::Observations)::Action
    if agent.model === nothing || agent.model.hof === nothing || length(agent.model.hof) == 0
        return rand(0:3)
    end
    num_actions = 3

    # 3 for testing Teke better one from pareto frontier
    best_eq = agent.model.hof.members[3].tree
    # For simplicity, we assume the equation takes no inputs and outputs an action index
    q = [best_eq(x, i) for i in 1:num_actions]

    # TODO sample
    action = round(Int32, argmax(q) - 1)  # assuming actions are 0-indexed  
    return action
    
end