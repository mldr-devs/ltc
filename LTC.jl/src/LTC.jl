module LTC
using SymbolicRegression
using PythonCall

function ltc_imports()
    return (jax=pyimport("jax"), jnp=pyimport("jax.numpy"), P=pyimport("ltc.symbolic.julia"))

end


function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    # print(options)
    return sum((prediction .- dataset.y) .^ 2) / dataset.n
end

mutable struct Env
    state::Py
    pysim::Py

    function Env(args::Py, key::Py=jax.random.key(0))
        (jax, jnp, P) = ltc_imports()
        key, k1, k2 = jax.random.split(key, 3)
        sim = P.Sim(args)
        c = sim.init(k1)
        new(c, sim)

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
    (jax, jnp, P) = ltc_imports()
    c, o = env.pysim.step(env.state, jnp.asarray(a))
    s_prev = pyconvert(Array{Float32,3}, pygetattr(env.state, "obs"))
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

end # module LTC
