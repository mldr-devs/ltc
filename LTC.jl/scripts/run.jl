
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = joinpath(@__DIR__, "..","..",".venv","bin","python")  # optional

import Pkg

Pkg.activate(joinpath(@__DIR__, ".."))  # activate LTC project
# Pkg.instantiate()
# Pkg.build("PythonCall")

import LTC
using PythonCall

LTC.f()

ltcsym = pyimport("ltc.symbolic")
print("LTC symbolic imported:", ltcsym)

jax = pyimport("jax")
jnp = pyimport("jax.numpy")
jeden=jnp.ones((3,3))
print("JAX jnp.ones:",jeden)
