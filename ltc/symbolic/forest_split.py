"""Random forest distilled from the train half of the agents.

The symbolic counterpart lives in ltc.symbolic.sr_split; the two read the same
split file but are fit independently, so either can be rerun on its own.
"""

import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ltc.symbolic.split import train_frame


def fit_forest_split(
    df_train: pd.DataFrame, n_estimators: int = 1500
) -> RandomForestClassifier:
    """Random forest on the same feature set as SR (agent id excluded).

    Class-balanced like fit_sr: 'balanced' reweights each sample by
    len(y) / (n_classes * count), the same formula, so neither model can win by
    always predicting the dominant action.
    """
    feat_cols = [c for c in df_train.columns if c not in {"agent", "action"}]
    X = df_train[feat_cols]
    y = df_train["action"]
    forest = RandomForestClassifier(
        n_estimators=n_estimators, oob_score=True, n_jobs=-1, class_weight="balanced"
    )
    forest.fit(X, y)
    return forest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit a random forest on the train half of the agents."
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
        help="Output path prefix; the forest is saved as <prefix>.split_forest.pkl",
    )
    parser.add_argument("--n_estimators", type=int, default=1500, help="RF trees")
    args = parser.parse_args()

    out_prefix = args.output or args.file.removesuffix(".csv")
    df_train = train_frame(pd.read_csv(args.file), args.split or f"{out_prefix}.split.json")

    print("Fitting random forest on the train half...")
    forest = fit_forest_split(df_train, n_estimators=args.n_estimators)
    print(f"RF OOB score: {forest.oob_score_:.4f}")

    forest_path = f"{out_prefix}.split_forest.pkl"
    joblib.dump(forest, forest_path)
    print(f"Saved: {forest_path}")
