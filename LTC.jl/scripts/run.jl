ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = joinpath(@__DIR__, "..", "..", ".venv", "bin", "python")  # optional

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))  # activate LTC project
using Revise

import LTC
using PythonCall

(jax, jnp, P) = LTC.ltc_imports()



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


    env = LTC.Env(args, key)
    sarsd = LTC.step!(env, ones(UInt32, 5))
    println(sarsd)



end


main()