""" Experimental bridge between julia and jax"""
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import os


# Konfiguracja JuliaCall - musi zostać wykonana PRZED importem juliacall
# 1. Użyj globalnego depot Julia (standardowa lokalizacja pakietów)
os.environ["JULIA_DEPOT_PATH"] = os.path.expanduser("~/.julia")

# 2. Aktywuj projekt z katalogu juliagents
os.environ["PYTHON_JULIACALL_PROJECT"] = os.path.expanduser("~/Documents/ml4wifi/ltc/juliagents")

# 3. Wskaż ścieżkę do wykonania Julia (używa globalnej instalacji 'julia' z PATH)
os.environ["PYTHON_JULIACALL_EXE"] = "julia"

# https://juliapy.github.io/PythonCall.jl/stable/
from juliacall import Main as jl

from juliacall import Main as jl







# https://juliapy.github.io/PythonCall.jl/stable/
from juliacall import Main as jl

# from pysr import PySRRegressor

from ltc.symbolic.regressor import Regressor

code='''
using Pkg
Pkg.activate("../.venv/julia_env")
# Pkg.add("MLJ")
# Pkg.add("SymbolicRegression")


using SymbolicRegression
using MLJ


X = 2randn(1000, 5)
y = @. 2*cos(X[:, 4]) + X[:, 1]^2 - 2

model = SRRegressor(
    binary_operators=[+, -, *, /],
    unary_operators=[cos],
    niterations=30
)
mach = machine(model, X, y)
f()=fit!(mach)
'''

if __name__ == '__main__':
    if True:
        jl.println("Hello from Julia!")
        jl.seval("f(x) = @. x^2 + 1")
        # jl.seval('using Pkg')
        # jl.seval('''Pkg.activate("../.venv/julia_env")''')
        # jl.seval("import Pkg; Pkg.instantiate()")
        print(jl.seval('Base.julia_cmd()'))
        print(jl.seval("Base.active_project()"))

        # jl.seval(code)
        x = jnp.asarray([1, 2, 3.])
        y = jl.f(jnp.asarray([1, 2, 3.]))
        print("Julia f(x):", y)

        def jax_f(x):
            return jax.experimental.io_callback(jl.f, jax.ShapeDtypeStruct(x.shape, x.dtype), x)

        yj = jax_f(x)
        print("JAX f(x):", yj)

        yj = jax.jit(jax_f)(x)
        print("JAX f(x):", yj)

        yj = jax.jit(jax.vmap(jax_f))(jnp.stack([x,x]))
        print("JAX f(x):", yj)

        @jax.custom_batching.custom_vmap
        def jax_f2(x):
            return jax.experimental.io_callback(jl.f, jax.ShapeDtypeStruct(x.shape, x.dtype), x)

        @jax_f2.def_vmap
        def jax_f2_vmap(axis_size, in_batched,x):
            ret = [jax.experimental.io_callback(jl.f, jax.ShapeDtypeStruct(_x.shape, _x.dtype), _x) for _x in x]
            return jnp.stack(ret, 0), True

        yj2 = jax.jit(jax.vmap(jax_f2))(jnp.stack([x,x]))
        print("JAX f2(x):", yj2)
    # os.exit(0)
    X = 2 * np.random.randn(100, 5)
    y = 2 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 2
    # model = PySRRegressor(
    #     binary_operators=["+", "*"],
    #     niterations=2
    # )
    # R = partial(PySRRegressor, binary_operators=["+", "*"], niterations=2)
    R= Regressor

    # models = [deepcopy(model), deepcopy(model)]
    models = [R(), R()]

    XX = jnp.stack([X, X])
    yy = jnp.stack([y, y])



    @jax.custom_batching.custom_vmap
    def fit_models(X, y):
        return jax.experimental.io_callback(pyfit, jax.ShapeDtypeStruct(y.shape, y.dtype), 0, X, y)


    def pyfit(i, X,y):
        print('Fitting model', i)
        m = models[i]
        m.fit(X, y)
        return jnp.zeros_like(y)

    @fit_models.def_vmap
    def fit_models_vmap(axis_size, in_batched, X, y):
        rets = []
        for i in range(axis_size):
            r = jax.experimental.io_callback(pyfit, jax.ShapeDtypeStruct(y[i].shape, y[i].dtype), i, X[i], y[i])
            rets.append(r)
        return jnp.stack(rets, 0), True

    def fit_models2(X, y):
        return jax.experimental.io_callback(pyfit, jax.ShapeDtypeStruct(y.shape, y.dtype), 0, X, y)

    jax.jit(jax.vmap(fit_models))(XX, yy)



