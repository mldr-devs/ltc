import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def fit_forest(df: pd.DataFrame, n_estimators: int = 1500) -> RandomForestClassifier:
    X = df.drop(columns=["action"])
    y = df["action"]
    # Class-balanced, matching fit_sr and fit_forest_split.
    forest = RandomForestClassifier(
        n_estimators=n_estimators, oob_score=True, n_jobs=-1, class_weight="balanced"
    )
    forest.fit(X, y)
    return forest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a random forest on a history CSV")
    parser.add_argument("--file", type=str, required=True, help="Path to .csv file")
    parser.add_argument(
        "--n_estimators", type=int, default=1500, help="Number of trees"
    )
    parser.add_argument("--output", type=str, default=None, help="Output .forest.pkl path")
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    forest = fit_forest(df, n_estimators=args.n_estimators)

    print(f"OOB score: {forest.oob_score_:.4f}")

    out_path = args.output if args.output else args.file.replace(".csv", ".forest.pkl")
    joblib.dump(forest, out_path)
    print(f"Saved: {out_path}")
