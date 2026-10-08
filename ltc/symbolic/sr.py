import argparse
import os
import pickle

from ltc.symbolic.util import SimplexCode

# os.environ.setdefault("PYTHON_JULIAPKG_EXE", "/opt/homebrew/bin/julia")

import numpy as np
import pandas as pd
from pysr import PySRRegressor


def fit_sr(
    df_ag: pd.DataFrame,
    label_codes: SimplexCode | None = None,
    n_iterations: int = 100,
    n_populations: int = 10,
    output_directory: str | None = None,
    balanced: bool = False,
) -> PySRRegressor:
    """Fit one symbolic expression against the simplex-coded action.

    ``balanced`` reweights each sample by ``len(y) / (n_classes * count)`` so the
    model cannot win by always predicting the dominant action. It is off by
    default, because the squared loss on simplex-coded labels is what makes the
    decoder work: its minimizer is ``E[y|x] = 2 p(x) - 1``, so an unweighted fit
    recovers the teacher's *probability* of transmitting and ``SimplexCode.probs``
    decodes it straight back. Reweighting moves that minimizer off the conditional
    mean and onto the decision boundary, which is how the bursty run ended up with
    ``f = 2 buffer_9 - 1``: a hard "transmit whenever the buffer is non-empty"
    where the teacher actually transmits with probability ~0.6. Ten stations
    sharing a persistence of 1.0 collide forever.

    Turn it on only when the argmax is the product and the probabilities are not.
    """
    exclude_cols = {"agent", "action"}
    feat_cols = [c for c in df_ag.columns if c not in exclude_cols]
    X = df_ag[feat_cols].astype(np.float32)
    # y = (2.0 * df_ag["action"].astype(np.float32) - 1.0).to_numpy()
    yi = df_ag["action"].astype(int).to_numpy()
    simplex_code = label_codes or SimplexCode(T=2) # Assuming binary actions; adjust T if more actions
    y = simplex_code.encode(yi)

    if balanced:
        classes, counts = np.unique(yi, return_counts=True)
        class_weight = {c: len(yi) / (len(classes) * n) for c, n in zip(classes, counts)}
        weights = np.array([class_weight[c] for c in yi], dtype=np.float32)
    else:
        weights = None

    model = PySRRegressor(
        niterations=n_iterations,
        populations=n_populations,
        binary_operators=["+", "*", "/", "-", "^"],
        unary_operators=["exp"],
        constraints={"^": (-1, 1), "exp": 3},
        # elementwise_loss="LogitMarginLoss()",
        temp_equation_file=output_directory is None,
        turbo=True,
        output_directory=output_directory,
    )
    model.fit(X, y, weights=weights)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit per-agent symbolic regression models on a history CSV"
    )
    parser.add_argument("--file", type=str, required=True, help="Path to .csv file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path prefix; per-agent files are saved as <prefix>_agent_<n>.sr.pkl",
    )
    parser.add_argument(
        "--agent", type=int, default=None, help="Single agent to fit (default: all)"
    )
    parser.add_argument("--n_iterations", type=int, default=100, help="PySR iterations")
    parser.add_argument(
        "--n_populations", type=int, default=10, help="PySR populations"
    )
    parser.add_argument(
        "--pysr_output_dir",
        type=str,
        default=None,
        help="Directory for PySR equation files (default: system temp)",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=False,
        help="Class-balance the sample weights. Off by default: it trades the "
        "conditional mean the simplex decoder needs for the decision boundary.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    out_prefix = args.output if args.output else args.file.replace(".csv", "")

    agents = [args.agent] if args.agent is not None else sorted(df["agent"].unique())
    label_codes = SimplexCode(T=df["action"].nunique())
    for ag in agents:
        print(f"Fitting agent {ag}...")
        model = fit_sr(
            df[df["agent"] == ag],
            label_codes=label_codes,
            n_iterations=args.n_iterations,
            n_populations=args.n_populations,
            output_directory=args.pysr_output_dir,
            balanced=args.balanced,
        )
        out_path = f"{out_prefix}_agent_{ag}.sr.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved: {out_path}")
        print(model)
