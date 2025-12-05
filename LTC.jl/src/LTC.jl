module LTC
using SymbolicRegression
using PythonCall

# P = pyimport("ltc.symbolic.julia")

# jax = pyimport("jax")
# jnp = pyimport("jax.numpy")


function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    # print(options)
    return sum((prediction .- dataset.y) .^ 2) / dataset.n
end

# mutable struct Env
#     state::Py
#     pysim::Py

#     function Env(args::Py, key::Py=jax.random.key(0))
#         sim = P.Sim(args)
#         c = sim.init(key)
#         new(c, sim)

#     end
# end

# struct SARSD
#     s::Array{Float32,3}
#     a::Array{Int64,1}
#     r::Array{Float32,1}
#     s2::Array{Float32,3}
#     done::Array{Bool,1}
# end

# function step!(env::Env, a::Vector{UInt32})
#     c, o = env.pysim.step(env.state, jnp.asarray(a))
#     s_prev = pyconvert(Array{Float32,3}, pygetattr(env.state, "obs"))
#     ret = SARSD(
#         s_prev,
#         a,
#         pyconvert(Array{Float32,1}, pygetattr(o, "rewards")),
#         pyconvert(Array{Float32,3}, pygetattr(o, "observations")),
#         pyconvert(Array{Bool,1}, pygetattr(o, "terminals")),
#     )
#     env.state = c
#     return ret

# end

end # module LTC
