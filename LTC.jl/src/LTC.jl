module LTC
# TODO: disable precompilation while developing, remove later
# __precompile__(false)
using SymbolicRegression
using PythonCall

export ltc_imports, Env, step!, SARSD, Observations, Action, Actions
export SRAgent, train!, take_action, default_options


_pymods = nothing

"""
    ltc_imports()

Returns a named tuple of necessary Python modules.
"""
function ltc_imports()
    global _pymods
    if _pymods !== nothing
        return _pymods
    end
    _pymods = (jax=pyimport("jax"), jnp=pyimport("jax.numpy"), P=pyimport("ltc.symbolic.julia"))
    return _pymods

end

"""
    Env(args::Py, key::Union{Py,Nothing}=nothing)
A wrapper around the LTC environment.
"""
mutable struct Env
    state::Py
    pysim::Py
    initial_state::Py

    function Env(args::Py, key::Union{Py,Nothing}=nothing)
        (jax, _, P) = ltc_imports()
        if key === nothing
            key = jax.random.key(0)
        end
        key, k1 = jax.random.split(key, 2)
        sim = P.Sim(args)
        c = sim.init(k1)
        new(c, sim, c)

    end
end

function reset!(env::Env)
    env.state = env.initial_state
end

const Observation = AbstractMatrix{Float32}
const Observations = AbstractArray{Float32,3}
const Action = Int32
const Actions = Array{Action,1}



"""
A struct representing a single step transition in the environment.
"""
struct Transition
    s::Observation
    a::Action
    r::Float32
    s2::Observation
    done::Bool
end




"""
    step!(env::Env, a::Actions) -> Tuple{AbstractVector{Transition}, Py}

Advances the environment by one step using the provided actions.

# Arguments
- `env::Env`: The environment to advance. This object will be mutated; its internal state will be updated to reflect the result of the step.
- `a::Actions`: A vector of actions to apply to the environment.

# Returns
- `transitions::AbstractVector{Transition}`: A vector of `Transition` objects, each representing the result of applying an action to the environment.
- `o::Py`: A Python object containing additional information about the step (e.g., observations, rewards, terminals).

# Side Effects
- Mutates `env.state` to the new state after the step.
"""
function step!(env::Env, a::Actions)::Tuple{AbstractVector{Transition}, Py}
    (_, jnp, _) = ltc_imports()
    c, o = env.pysim.step(env.state, jnp.asarray(a))
    s = pyconvert(Observations, env.state.obs)
    s2 = pyconvert(Observations, o.observations)
    r = pyconvert(Vector{Float32}, o.rewards)
    done = pyconvert(Vector{Bool}, o.terminals) 
    ret = [Transition(
        s[i, :, :],
        a[i],
        r[i],
        s2[i, :, :],
        done[i],
    ) for i in eachindex(a)]
    
    env.state = c
    return ret,o

end

include("agent.jl")

end # module LTC
