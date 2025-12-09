ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = joinpath(@__DIR__, "..", "..", ".venv", "bin", "python")  # optional

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))  # activate LTC project
using Revise

using LTC
using PythonCall



function main()
    (jax, jnp, P) = ltc_imports()
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

    obs = pyconvert(Observations, obs)
    n = pyconvert(Int, args.n)
    n_drl = pyconvert(Int, args.n_drl)
    n_epochs = pyconvert(Int, args.n_epochs)
    n_steps = pyconvert(Int, args.n_steps)
    window_size = pyconvert(Int, args.window_size)
    seed = pyconvert(Int, args.seed)
    traffic_type = pyconvert(String, args.traffic_type)   


    env = Env(args, key)

    agents = [SRAgent(default_options(), 1000) for _ in 1:n_drl]
    
    for step in 1:n_steps
        a = [take_action(ag, ob) for (ag, ob) in zip(agents, eachslice(obs, dims=1))]  
        transitions = step!(env, a)
        for i in eachindex(agents)
            train!(agents[i], transitions[i])
        end

        println(transitions)
        if step > 10
            break # TODO remove
        end
    end
    

end

main()