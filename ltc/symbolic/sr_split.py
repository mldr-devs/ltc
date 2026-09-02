"""Symbolic model distilled from the train half of the agents.

The random-forest counterpart lives in ltc.symbolic.forest_split; the two read
the same split file but are fit independently, so either can be rerun on its own.
"""

import argparse
import os
import pickle

from ltc.symbolic.sr import fit_sr
from ltc.symbolic.split import train_frame
from ltc.symbolic.util import SimplexCode

os.environ.setdefault("PYTHON_JULIAPKG_EXE", "/opt/homebrew/bin/julia")

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit a symbolic model on the train half of the agents, "
        "leaving the other half as a held-out evaluation set."
    )
    parser.add_argument("--file", type=str, required=True, help="Path to .csv file")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split .json from ltc.symbolic.split. Defaults to <prefix>.split.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path prefix; the model is saved as <prefix>.split_sr.pkl",
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
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    out_prefix = args.output or args.file.removesuffix(".csv")
    df_train = train_frame(df, args.split or f"{out_prefix}.split.json")

    print("Fitting symbolic regression on the train half...")
    sr_model = fit_sr(
        df_train,
        label_codes=SimplexCode(T=df["action"].nunique()),
        n_iterations=args.n_iterations,
        n_populations=args.n_populations,
        output_directory=args.pysr_output_dir,
    )

    sr_path = f"{out_prefix}.split_sr.pkl"
    with open(sr_path, "wb") as f:
        pickle.dump(sr_model, f)
    print(f"Saved: {sr_path}")
    print(sr_model)
