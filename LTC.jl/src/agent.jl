import SymbolicRegression: Options, equation_search, HallOfFame
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
    state::Union{Nothing,AbstractVector}
    hof::Union{Nothing,HallOfFame}
end



mutable struct SRAgent
    model::Union{Nothing,SymQ}
    target_model::Union{Nothing,SymQ}
    options::Options
    replay_buffer::CircularBuffer{Transition}
    step::Int64

    function SRAgent(options::Options, replay_buffer_capacity::Int)
        new(nothing, nothing, options, CircularBuffer{Transition}(replay_buffer_capacity), 0)
    end
end

function train!(agent::SRAgent, transition::Transition, batch_size::Int=2)
    push!(agent.replay_buffer, transition)
    if length(agent.replay_buffer) < batch_size
        return
    end

    batch =  rand(agent.replay_buffer, batch_size)
    X = hcat([make_X(sarsd.s, sarsd.a) for sarsd in batch]...)
    # @info "Training on batch X size: $(size(X))"
    

    y = randn(Float32, batch_size)

    if agent.model === nothing
        state, hof = equation_search(X, y; options=agent.options, niterations=1, return_state=true)
    else
        state, hof = equation_search(X, y; options=agent.options, niterations=1, return_state=true, saved_state=(agent.model.state, agent.model.hof))
    end
    agent.model = SymQ(state, hof)


    # TODO
    agent.step += 1

    if agent.step % 10 == 0
        agent.target_model = deepcopy(agent.model)
    end
end

function make_X(obs::Observation, action::Action, n_actions=3)::AbstractMatrix{Float32}
    hot1_act = zeros(eltype(obs), n_actions)  
    hot1_act[action + 1] = 1  # assuming actions are 0-indexed
    X = [obs[:]; hot1_act]
    return reshape(X, :, 1)
    
end

function take_action(agent::SRAgent, observations::Observation)::Action
    num_actions = 3
    if agent.model === nothing || agent.model.hof === nothing || length(agent.model.hof.members) == 0
        return rand(0:num_actions-1)
    end
    

    # 3 for testing Teke better one from pareto frontier
    best_eq = agent.model.hof.members[3].tree
    
    # X = randn(2, 100)

    q = [best_eq(make_X(observations, Int32(a))) for a in 0:num_actions-1]

    # TODO sample
    action = round(Int32, argmax(q) - 1)  # assuming actions are 0-indexed  
    return action

end