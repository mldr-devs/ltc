"""Symbolic model distilled from the train half of the agents.

The random-forest counterpart lives in ltc.symbolic.forest_split; the two read
the same split file but are fit independently, so either can be rerun on its own.
"""

import argparse
import json
import os
import pickle

import numpy as np

from ltc.symbolic.calibration import ScaleCalibrator
from ltc.symbolic.sr import fit_sr
from ltc.symbolic.split import train_frame
from ltc.symbolic.util import SimplexCode

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
    parser.add_argument(
        "--coding",
        type=str,
        default="simplex",
        choices=["simplex", "logit"],
        help="How the expression encodes the action. 'logit' fits T-1 logits against "
        "a zero reference class under a cross-entropy loss; 'simplex' fits a simplex "
        "codeword under squared loss. Recorded in the sidecar, so ltc.run decodes to "
        "match.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=False,
        help="Class-balance the sample weights. Off by default; see ltc.symbolic.sr.fit_sr "
        "for why the unweighted fit is what the simplex decoder needs.",
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
        balanced=args.balanced,
        coding=args.coding,
    )

    sr_path = f"{out_prefix}.split_sr.pkl"
    with open(sr_path, "wb") as f:
        pickle.dump(sr_model, f)
    print(f"Saved: {sr_path}")
    print(sr_model)

    scale_path = f"{out_prefix}.split_sr.scale.json"
    T = df["action"].nunique()

    if args.coding == "logit":
        # Softmax is a probability by construction, so there is nothing to calibrate.
        with open(scale_path, "w") as f:
            json.dump({"coding": "logit", "T": T}, f, indent=2)
        print(f"Saved: {scale_path}")
    else:
        # The decoded probabilities are too flat whenever the expression under-fits
        # (see ltc.symbolic.calibration), so fit the sharpening scalar here, on the
        # same train half, and leave it beside the model.
        feat_cols = [c for c in df_train.columns if c not in {"agent", "action"}]
        fx = np.asarray(sr_model.predict(df_train[feat_cols].astype(np.float32)))
        calibrator = ScaleCalibrator(T=T).fit(fx.reshape(len(df_train), T - 1),
                                              df_train["action"].astype(int).to_numpy())
        with open(scale_path, "w") as f:
            json.dump({"coding": "simplex", "scale": calibrator.scale, "T": T,
                       "clipped_fraction": calibrator.clipped_fraction_}, f, indent=2)
        print(f"Calibrated scale a={calibrator.scale:.4g} "
              f"(rows clipped onto the simplex: {calibrator.clipped_fraction_:.3f})")
        if calibrator.clipped_fraction_ > 0.9:
            print("Warning: nearly every row saturates at this scale, so --stochastic_policy "
                  "degenerates to the argmax. Override with --sr_scale if that is not wanted.")
        print(f"Saved: {scale_path}")
