from functools import partial

import sympy
from sympy.stats import Normal
from pysr import PySRRegressor

Regressor = partial(PySRRegressor, maxsize=20, niterations=10,
                    # < Increase me for better results
                    binary_operators=["+", "*", "^"],
                    unary_operators=["exp", "inv(x) = 1/x",
                                     'r(x)=randn(eltype(x),size(x)...)'
                                     # ^ Custom operator (julia syntax)
                                     ],
                    extra_sympy_mappings={"inv": lambda x: 1 / x,
                                          "r": lambda x: Normal(name='N', mean=0, std=1)},
                    # ^ Define operator for SymPy as well
                    elementwise_loss="loss(prediction, target) = sum((prediction - target)^2)",
                    constraints={'r': 1, '^': (-1, 1)},
                    parallelism='serial',
                    input_stream="devnull",

                    )
