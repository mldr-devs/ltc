import numpy as np
import pandas as pd
from pysr import PySRRegressor

from .nn2sym import Target

if __name__ == '__main__':
    target = Target("history_5_5_42.pkl.lz4")
    observations, qvals = target()

    fo = np.squeeze(observations)
    hdf = pd.DataFrame(fo,
                       columns='buffer,channel,ret_c,no_tx,batter'.split(','))


    model = PySRRegressor(
        maxsize=20,
        niterations=40,  # < Increase me for better results
        binary_operators=["+", "*", "^"],
        unary_operators=[
            "exp",
            "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = sum((prediction - target)^2)",
        # ^ Custom loss function (julia syntax)
    )

    model.fit(hdf, qvals)
    print(model)