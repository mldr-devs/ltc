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

    function Env(args::Py, key::Union{Py,Nothing}=nothing)
        (jax, _, P) = ltc_imports()
        if key === nothing
            key = jax.random.key(0)
        end
        key, k1 = jax.random.split(key, 2)
        sim = P.Sim(args)
        c = sim.init(k1)
        new(c, sim)

    end
end

const Observation = AbstractMatrix{Float32}
const Observations = AbstractArray{Float32,3}
const Action = Int32
const Actions = Array{Action,1}

"""
    SARSD
A struct representing a single step transition in the environment.
"""
struct SARSD
    s::Observations
    a::Actions
    r::Array{Float32,1}
    s2::Observations
    done::Array{Bool,1}
end

"""
Advances the environment by one step.
"""
function step!(env::Env, a::Actions)::SARSD
    (_, jnp, _) = ltc_imports()
    c, o = env.pysim.step(env.state, jnp.asarray(a))
    s_prev = pyconvert(Array{Float32,3}, env.state.obs)
    ret = SARSD(
        s_prev,
        a,
        pyconvert(Array{Float32,1}, o.rewards),
        pyconvert(Array{Float32,3}, o.observations),
        pyconvert(Array{Bool,1}, o.terminals),
    )
    env.state = c
    return ret

end

include("agent.jl")

end # module LTC
