ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = joinpath(@__DIR__, "..", "..", ".venv", "bin", "python")  # optional

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))  # activate LTC project
using Revise

using LTC
using PythonCall
using Base.Threads


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
    num_actions = pyconvert(Int, num_actions)


    env = Env(args, key)

    agents = [SRAgent(default_options(), 1000, num_actions) for _ in 1:n_drl]
    rewards = zeros(Float32, n, n_steps)

    for ep in 1:n_epochs
        @info "Starting new epoch"  epoch=ep 

        for step in 1:n_steps
            @info "Step" step=step
            a = [take_action(ag, ob) for (ag, ob) in zip(agents, eachslice(obs, dims=1))]
            transitions = step!(env, a)
            @threads for i in eachindex(agents)
                train!(agents[i], transitions[i], batch_size=128)
            end

            # println(transitions)
            if step > 1000
                break # TODO remove
            end
            rewards[:, step] = [t.r for t in transitions]
        end
        reset!(env)
    end


end

main()