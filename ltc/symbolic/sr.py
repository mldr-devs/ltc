import argparse
import os
import pickle

os.environ.setdefault("PYTHON_JULIAPKG_EXE", "/opt/homebrew/bin/julia")

import numpy as np
import pandas as pd
from pysr import PySRRegressor


def fit_sr(
    df_ag: pd.DataFrame,
    n_iterations: int = 100,
    n_populations: int = 10,
    output_directory: str | None = None,
) -> PySRRegressor:
    exclude_cols = {"agent", "action"}
    feat_cols = [c for c in df_ag.columns if c not in exclude_cols]
    X = df_ag[feat_cols].astype(np.float32)
    y = (2.0 * df_ag["action"].astype(np.float32) - 1.0).to_numpy()

    model = PySRRegressor(
        niterations=n_iterations,
        populations=n_populations,
        binary_operators=["+", "*", "/", "-", "^"],
        unary_operators=["exp"],
        constraints={"^": (-1, 1), "exp": 3},
        elementwise_loss="LogitMarginLoss()",
        temp_equation_file=True,
        turbo=True,
        output_directory=output_directory,
    )
    model.fit(X, y)
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
        "--agent", type=int, default=0, help="Single agent to fit (default: 0)"
    )
    parser.add_argument("--n_iterations", type=int, default=100, help="PySR iterations")
    parser.add_argument(
        "--n_populations", type=int, default=10, help="PySR populations"
    )
    parser.add_argument(
        "--pysr_output_dir", type=str, default=None,
        help="Directory for PySR equation files (default: system temp)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    out_prefix = args.output if args.output else args.file.replace(".csv", "")

    agents = [args.agent] if args.agent is not None else sorted(df["agent"].unique())
    for ag in agents:
        print(f"Fitting agent {ag}...")
        model = fit_sr(
            df[df["agent"] == ag],
            n_iterations=args.n_iterations,
            n_populations=args.n_populations,
            output_directory=args.pysr_output_dir,
        )
        out_path = f"{out_prefix}_agent_{ag}.sr.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved: {out_path}")
        print(model)
