ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = joinpath(@__DIR__, "..", "..", ".venv", "bin", "python")  # optional

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))  # activate LTC project
using Revise

import LTC
using PythonCall

P = pyimport("ltc.symbolic.julia")

jax = pyimport("jax")
jnp = pyimport("jax.numpy")

mutable struct Env
    state::Py
    pystep::Py

    function Env(args::Py, key::Py=jax.random.key(0))
        key, k1, k2 = jax.random.split(key, 3)
        sim = P.Sim(args)
        c = sim.init(k1)
        new(c, sim.step)

    end
end

struct SARSD
    s::Array{Float32,3}
    a::Array{Int64,1}
    r::Array{Float32,1}
    s2::Array{Float32,3}
    done::Array{Bool,1}
end

function step!(env::Env, a::Vector{UInt32})
    c, o = env.pystep(env.state, jnp.asarray(a))
    s_prev = pyconvert(Array{Float32,3}, pygetattr(env.state, "obs"))
    ret = SARSD(
        s_prev,
        a,
        pyconvert(Array{Float32,1}, pygetattr(o, "rewards")),
        pyconvert(Array{Float32,3}, pygetattr(o, "observations")),
        pyconvert(Array{Bool,1}, pygetattr(o, "terminals")),
    )
    env.state = c
    return ret

end



function main()
    args = P.setup_args()

    n = args.n
    n_drl = args.n_drl
    n_epochs = args.n_epochs
    n_steps = args.n_steps
    window_size = args.window_size
    seed = args.seed
    traffic_type = args.traffic_type

    key = jax.random.key(seed)
    num_actions = P._len(P.Actions)
    actions = jnp.zeros(n, dtype=jnp.int32)
    buffer_states = jnp.zeros(n, dtype=jnp.int32)
    power_states = jnp.full(n, P.INITIAL_CAPACITY, dtype=jnp.int32)
    channel_state = 0
    obs = P.make_obs(window_size, n)
    rewards = jnp.zeros(n)
    terminals = jnp.full(n, false, dtype=jnp.bool)


    env = Env(args, key)
    sarsd = step!(env, ones(UInt32, 5))
    println(sarsd)



end


main()