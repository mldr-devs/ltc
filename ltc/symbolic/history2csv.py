import argparse
import itertools as it

import cloudpickle
import jax.numpy as jnp
import lz4.frame
import numpy as np
import pandas as pd

from ltc.sim.constants import ObsIdx
from ltc.sim.process_output import normalize_obs
from ltc.utils.history import unpack_history

FEATURE_NAMES = [name.lower() for name, _ in sorted(
    ((k, v) for k, v in vars(ObsIdx).items() if k.isupper()), key=lambda kv: kv[1]
)]


def build_column_names(window_size: int) -> list[str]:
    return [f"{f}_{t}" for t, f in it.product(range(window_size), FEATURE_NAMES)]


def decision_frame(history, meta, epochs: int = 1) -> pd.DataFrame:
    """One row per DRL decision: the normalized observation it acted on and the raw action it chose."""
    n = max(meta['n'], meta.get('n_final') or meta['n'])
    n_drl = meta.get('n_drl') or n
    observations = np.asarray(history.observations)
    keep = len(observations) if epochs == 0 else min(epochs, len(observations))
    observations = observations[-keep:]
    actions = np.asarray(history.actions)[-keep:]
    decisions = np.asarray(history.decisions)[-keep:]
    window_size = observations.shape[-2]

    X, y, agent = [], [], []
    for e in range(keep):
        obs = np.asarray(normalize_obs(jnp.asarray(observations[e, :-1, :n_drl]), meta['queue_size'], meta['tx_slots']))
        for a in range(n_drl):
            mask = decisions[e, 1:, a]
            X.append(obs[mask, a].reshape(mask.sum(), -1))
            y.append(actions[e, 1:, a][mask])
            agent.append(np.full(mask.sum(), a))

    df = pd.DataFrame(np.concatenate(X), columns=build_column_names(window_size))
    df["agent"] = np.concatenate(agent)
    df["action"] = np.concatenate(y)
    return df.sort_values("agent", kind="stable").reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten a v2 history into the (observation, action) distillation dataset")
    parser.add_argument("--file", type=str, required=True, help="Path to history .pkl.lz4 file")
    parser.add_argument("--output", type=str, default=None, help="Output .csv path")
    parser.add_argument("--epochs", type=int, default=1, help="Trailing epochs with observations to pool (0 = all recorded).")
    parser.add_argument("--labels", type=str, default="actions", choices=["actions"], help="Label source: the actions the agent took.")
    args = parser.parse_args()

    print(f"Loading: {args.file}")
    with lz4.frame.open(args.file, "rb") as f:
        _, history, meta = unpack_history(cloudpickle.load(f))
    if history.observations is None:
        raise ValueError("The history has no observations; rerun training without dropping them.")

    df = decision_frame(history, meta, args.epochs)
    out_path = args.output or args.file.replace(".pkl.lz4", ".csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} decisions, {df['agent'].nunique()} agents, "
          f"{df['action'].nunique()} distinct actions)")
