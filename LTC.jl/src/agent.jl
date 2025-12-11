import SymbolicRegression: Options, equation_search, HallOfFame
using Random, DataStructures, StatsBase
using SymbolicRegression.MLJInterfaceModule: choose_best
using SymbolicRegression.HallOfFameModule: format_hall_of_fame

"""Noise function for SR"""
r(x) = randn(eltype(x), size(x)...)

function default_options()
    Options(
        binary_operators=[+, *, /, -],
        unary_operators=[cos, exp, sin, r],
        populations=20,
        save_to_file=false,
        verbosity=0,
    )
end

struct SymQ
    state::Union{Nothing,AbstractVector}
    hof::Union{Nothing,HallOfFame}
end

function best_equation(model::SymQ, options::Options)
    trees, scores, losses, complexities = format_hall_of_fame(model.hof, options)
    best_idx = choose_best(trees=trees, losses=losses, scores=scores, complexities=complexities, options=options)
    return trees[best_idx]
end

function predict(model::SymQ, X::AbstractMatrix{Float32}, options::Options)::Vector{Float32}

    best_eq = best_equation(model, options)
    return best_eq(X)
end


mutable struct SRAgent
    model::Union{Nothing,SymQ}
    equation::Union{Nothing,Any}
    target_equation::Union{Nothing,Any}

    options::Options
    replay_buffer::CircularBuffer{Transition}
    step::Int64
    epsilon::Float32
    gamma::Float32
    num_actions::Int

    function SRAgent(options::Options, replay_buffer_capacity::Int, num_actions::Int=3, epsilon::Float32=1.0f0, gamma::Float32=0.99f0)
        new(nothing, nothing,nothing, options, CircularBuffer{Transition}(replay_buffer_capacity), 0, epsilon, gamma, num_actions)
    end
end

function train!(agent::SRAgent, transition::Transition; batch_size::Int=2)
    push!(agent.replay_buffer, transition)

    function common!(agent::SRAgent)
        agent.epsilon = max(0.1f0, agent.epsilon * 0.995f0)
        agent.step += 1
    end

    if length(agent.replay_buffer) < batch_size
        return
    end

    batch = rand(agent.replay_buffer, batch_size)
    X = hcat([make_X(sarsd.s, sarsd.a) for sarsd in batch]...)
    y = make_target(agent, batch)

    if agent.model === nothing
        state, hof = equation_search(X, y; options=agent.options, niterations=1, return_state=true, parallelism=:serial)
    else
        state, hof = equation_search(X, y; options=agent.options, niterations=1, return_state=true, saved_state=(agent.model.state, agent.model.hof), parallelism=:serial)
    end
    agent.model = SymQ(state, hof)
    agent.equation = best_equation(agent.model, agent.options)


    common!(agent)


    if agent.step % 10 == 0 || agent.target_equation === nothing
        agent.target_equation = deepcopy(agent.equation)
    end
end


function make_X(obs::Observation, action::Action, n_actions=3)::AbstractMatrix{Float32}
    hot1_act = zeros(eltype(obs), n_actions)
    hot1_act[action+1] = 1  # assuming actions are 0-indexed
    X = [obs[:]; hot1_act]
    return reshape(X, :, 1)

end

function take_action(agent::SRAgent, observations::Observation)::Action
    if agent.model === nothing || agent.model.hof === nothing || length(agent.model.hof.members) == 0
        return rand(0:agent.num_actions-1)
    end

    q = [agent.equation(make_X(observations, Int32(a)))[1] for a in 0:agent.num_actions-1]

    mq = maximum(q)
    weights = [x == mq ? 1.0f0 : 0.0f0 for x in q]

    action = sample(1:agent.num_actions, Weights(weights)) - 1

    # Explore
    if rand() < agent.epsilon
        action = rand(0:agent.num_actions-1)
    end
    return action

end

function make_target(agent::SRAgent, batch::AbstractVector{Transition})::Vector{Float32}
    n = length(batch)
    targets = Vector{Float32}(undef, n)
    if agent.target_equation === nothing
        # Alternative option: use immediate rewards only
        # for i in 1:n
        #     targets[i] = batch[i].r
        # end
        return r(targets)
    end

    best_eq = agent.target_equation

    for i in 1:n
        sarsd = batch[i]
        if sarsd.done
            targets[i] = sarsd.r
        else
            max_q_next = -Inf32
            for a in 0:agent.num_actions-1
                q_next = best_eq(make_X(sarsd.s2, Int32(a)))[1] # vector output
                if q_next > max_q_next
                    max_q_next = q_next
                end
            end

            targets[i] = sarsd.r + agent.gamma * max_q_next
        end
    end
    return targets

end