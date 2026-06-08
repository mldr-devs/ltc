import argparse
import json
import os
import pickle

import joblib

from ltc.symbolic.sr import fit_sr
from ltc.symbolic.util import SimplexCode

os.environ.setdefault("PYTHON_JULIAPKG_EXE", "/opt/homebrew/bin/julia")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def split_agents(agents: list[int]) -> tuple[list[int], list[int]]:
    """Split the sorted agent ids into a train half and a held-out test half."""
    agents = sorted(agents)
    half = len(agents) // 2
    return agents[:half], agents[half:]


def fit_forest_split(
    df_train: pd.DataFrame, n_estimators: int = 1500
) -> RandomForestClassifier:
    """Random forest on the same feature set as SR (agent id excluded)."""
    feat_cols = [c for c in df_train.columns if c not in {"agent", "action"}]
    X = df_train[feat_cols]
    y = df_train["action"]
    forest = RandomForestClassifier(
        n_estimators=n_estimators, oob_score=True, n_jobs=-1
    )
    forest.fit(X, y)
    return forest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit a single SR model (and a RF baseline) on half of the "
        "agents, leaving the other half as a held-out evaluation set."
    )
    parser.add_argument("--file", type=str, required=True, help="Path to .csv file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path prefix; artifacts saved as <prefix>.split_*",
    )
    parser.add_argument("--n_iterations", type=int, default=100, help="PySR iterations")
    parser.add_argument(
        "--n_populations", type=int, default=10, help="PySR populations"
    )
    parser.add_argument(
        "--n_estimators", type=int, default=1500, help="RF trees"
    )
    parser.add_argument(
        "--pysr_output_dir",
        type=str,
        default=None,
        help="Directory for PySR equation files (default: system temp)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    out_prefix = args.output if args.output else args.file.replace(".csv", "")

    agents = sorted(df["agent"].unique().tolist())
    train_agents, test_agents = split_agents(agents)
    print(f"Train agents: {train_agents}")
    print(f"Held-out test agents: {test_agents}")

    df_train = df[df["agent"].isin(train_agents)]
    label_codes = SimplexCode(T=df["action"].nunique())

    print("Fitting symbolic regression on the train half...")
    sr_model = fit_sr(
        df_train,
        label_codes=label_codes,
        n_iterations=args.n_iterations,
        n_populations=args.n_populations,
        output_directory=args.pysr_output_dir,
    )

    print("Fitting random forest baseline on the train half...")
    forest = fit_forest_split(df_train, n_estimators=args.n_estimators)
    print(f"RF OOB score: {forest.oob_score_:.4f}")

    sr_path = f"{out_prefix}.split_sr.pkl"
    forest_path = f"{out_prefix}.split_forest.pkl"
    split_path = f"{out_prefix}.split.json"

    with open(sr_path, "wb") as f:
        pickle.dump(sr_model, f)
    joblib.dump(forest, forest_path)
    with open(split_path, "w") as f:
        json.dump(
            {"train_agents": train_agents, "test_agents": test_agents}, f, indent=2
        )

    print(f"Saved: {sr_path}")
    print(f"Saved: {forest_path}")
    print(f"Saved: {split_path}")
    print(sr_model)
