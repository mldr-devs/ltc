import argparse
import functools as ft
import itertools as it

import cloudpickle
import jax
import jax.numpy as jnp
import lz4.frame
import numpy as np
import pandas as pd

from ltc.agents import QNetwork
from ltc.utils.history import resolve_history_file, unpack_history

FEATURE_NAMES = ["buffer", "channel", "ret_c", "no_tx", "action"]

_model = QNetwork(num_actions=2, num_layers=1, dim=64, num_heads=4)


def build_column_names(window_size: int) -> list[str]:
    return [f"{f}_{t}" for t, f in it.product(range(window_size), FEATURE_NAMES)]


@jax.jit
def predict_one(params, state, obs, key):
    keys = map(ft.partial(jax.random.fold_in, key), it.count())
    rngs = {"params": next(keys), "dropout": next(keys), "rlib": next(keys)}
    q_vals, _ = _model.apply(
        {"params": params, **state}, obs, training=False, rngs=rngs, mutable=["loss"]
    )
    return q_vals


@jax.jit
def compute_qvals(params, state, observations, key):
    return jax.vmap(predict_one, in_axes=(0, 0, 1, None))(params, state, observations, key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Q-network observations and Q-values from history file"
    )
    parser.add_argument("--file", type=str, help="Path to history .pkl.lz4 file")
    parser.add_argument(
        "--n", type=int, default=5, help="Number of agents used for filename matching"
    )
    parser.add_argument(
        "--n_drl",
        type=int,
        default=5,
        help="Number of DRL agents used for filename matching",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed used for filename matching"
    )

    args = parser.parse_args()

    history_file = resolve_history_file(
        n=args.n, n_drl=args.n_drl, seed=args.seed, file_path=args.file
    )

    print(f"Loading: {history_file}")
    with lz4.frame.open(history_file, "rb") as f:
        payload = cloudpickle.load(f)

    drl_state, history, _ = unpack_history(payload)
    params, state = drl_state.params, drl_state.net_state
    n_agents = jax.tree.leaves(params)[0].shape[0]

    # Last epoch: shape [n_steps, n_agents, window_size, n_features]
    observations = history.observations[-1, ...]
    n_steps, _, window_size, _ = observations.shape

    key = jax.random.key(42)
    qvals = compute_qvals(
        params, state, observations, key
    )  # [n_agents, n_steps, n_actions]

    # Build flat feature matrix: [n_agents * n_steps, window_size * n_features]
    # transpose [n_steps, n_agents, w, f] -> [n_agents, n_steps, w, f] then flatten last two dims
    XX = np.asarray(observations).transpose(1, 0, 2, 3).reshape(n_agents * n_steps, -1)
    actions = np.asarray(qvals.argmax(axis=-1)).flatten()  # [n_agents * n_steps]
    agent_ids = np.repeat(np.arange(n_agents), n_steps)

    df = pd.DataFrame(XX, columns=build_column_names(window_size))
    df["agent"] = agent_ids
    df["action"] = actions

    out_path = str(history_file).replace(".pkl.lz4", ".csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")
