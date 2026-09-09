import os
import pickle
import unittest

import cloudpickle
import jax
import lz4.frame
import numpy as np
import pandas as pd
import sympy

# Artifacts of a one-off manual run, from before the pipeline distilled per-agent
# models. Nothing regenerates them, so the test only runs where they still exist.
BASE = "mid_traffic_DDQN_history_10_10_1"
AGENT = 5
CSV_FILE = f"out/{BASE}.csv"
MODEL_FILE = f"out/{BASE}_agent_{AGENT}.sr.pkl"


@unittest.skipUnless(
    os.path.exists(CSV_FILE) and os.path.exists(MODEL_FILE),
    f"test data not found: {CSV_FILE}, {MODEL_FILE}",
)
class LabelTest(unittest.TestCase):
    def test_label(self):
        agent, path = AGENT, MODEL_FILE

        df = pd.read_csv(CSV_FILE)
        with open(path, "rb") as _fh:
            model = pickle.load(_fh)

        _feat_cols = [c for c in df.columns if c not in {"agent", "action"}]

        expr = model.sympy()
        free_syms = sorted(expr.free_symbols, key=lambda s: s.name)
        f = sympy.lambdify(free_syms, expr, "numpy")
        yhat_sym = f(*[df[s.name].values for s in free_syms])
        y_hat = model.predict(df[_feat_cols].values, 8)
        self.assertTrue(np.all(y_hat == yhat_sym))
        where_agent = df["agent"] == agent
        y = df["action"][where_agent].values
        np.mean((y_hat[where_agent] > 0) == y)

        dfa = df[df["agent"] == agent]

        X = dfa[_feat_cols].astype(np.float32)
        yt = (2.0 * dfa["action"].astype(np.float32) - 1.0).to_numpy()

        # from pysr import PySRRegressor
        # modelf = PySRRegressor(
        #     niterations=100,
        #     populations=10,
        #     binary_operators=["+", "*", "/", "-", "^"],
        #     unary_operators=["exp"],
        #     constraints={"^": (-1, 1), "exp": 3},
        #     elementwise_loss="LogitMarginLoss()",

        #     turbo=True,

        # )
        # modelf.fit(X, yt)

        # ((modelf.predict(X,4)>0)==y).mean()
        # ((modelf.predict(X, 5) > 0) == y).mean()

        # jf = modelf.jax()

        # jhat = jax.jit(jf['callable'])(X.values, jf['parameters'])
        # ((jhat>0)==y).mean()
        ...
