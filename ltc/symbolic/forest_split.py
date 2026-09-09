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
    df_train: pd.DataFrame, n_estimators: int = 1500, balanced: bool = False
) -> RandomForestClassifier:
    """Random forest on the same feature set as SR (agent id excluded).

    ``balanced`` sets class_weight='balanced', reweighting each sample by
    len(y) / (n_classes * count). It is off by default for the reason ltc.symbolic
    .sr.fit_sr gives: the replay samples its action from the forest's vote
    fractions, so those fractions have to be probabilities, and reweighting moves
    them off the conditional mean towards the decision boundary.

    Measured rather than argued. Eight forests over the bursty run, a full grid of
    {50, 1500} trees x {balanced, unweighted} x {ten pooled epochs, the last epoch
    alone}, each replayed for 3000 steps:

        trees  weights     data     throughput
           50  balanced  pooled         0.0000
         1500  balanced  pooled         0.0000
           50  none      pooled         0.0727
         1500  none      pooled         0.0700
           50  balanced  last           0.0020
         1500  balanced  last           0.0053
           50  none      last           0.0747
         1500  none      last           0.0747

    Every balanced fit deadlocks the network and every unweighted one does not,
    whatever the tree count or the dataset. Tree count does nothing: 50 and 1500
    agree to the third decimal in all four cells.

    The effect on the training distribution is small -- mean p(TX) in a congested
    state is 0.143 unweighted against 0.178 balanced -- and that is the point. What
    decides the replay is behaviour in states the teacher never visits, so a
    difference of 0.035 on the teacher's own rows separates a working policy from
    one that collides forever. It cannot be read off the fit; only a replay shows
    it, which is what ltc.symbolic.sr_select does for the symbolic path and nothing
    yet does for this one.
    """
    feat_cols = [c for c in df_train.columns if c not in {"agent", "action"}]
    X = df_train[feat_cols]
    y = df_train["action"]
    forest = RandomForestClassifier(
        n_estimators=n_estimators, oob_score=True, n_jobs=-1,
        class_weight="balanced" if balanced else None,
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
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=False,
        help="Class-balance the forest. Off by default: it deadlocked every replay it "
        "was measured in, whatever the tree count. See fit_forest_split.",
    )
    args = parser.parse_args()

    out_prefix = args.output or args.file.removesuffix(".csv")
    df_train = train_frame(pd.read_csv(args.file), args.split or f"{out_prefix}.split.json")

    print("Fitting random forest on the train half...")
    forest = fit_forest_split(df_train, n_estimators=args.n_estimators, balanced=args.balanced)
    print(f"RF OOB score: {forest.oob_score_:.4f}")

    forest_path = f"{out_prefix}.split_forest.pkl"
    joblib.dump(forest, forest_path)
    print(f"Saved: {forest_path}")
