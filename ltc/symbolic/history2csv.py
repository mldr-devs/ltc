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
from ltc.sim.process_output import normalize_obs
from ltc.symbolic.util import history_reshape
from ltc.utils.history import resolve_history_file, unpack_history

FEATURE_NAMES = ["buffer", "channel", "ret_c", "no_tx", "action_tx", "action_cs"]

_model = QNetwork(num_actions=2, num_layers=2, dim=64)


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
    parser.add_argument("--output", type=str, default=None, help="Output .csv path")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="How many trailing epochs to pool into the dataset (0 = all). One epoch "
        "is the converged policy but carries almost no congested states: on the bursty "
        "run its last epoch holds 207 buffer-full steps and 26 with ret_c > 0, far too "
        "few to fit the teacher's backoff, so the distilled policy transmits "
        "unconditionally and every station locks into a permanent collision.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="qvals",
        choices=["qvals", "actions"],
        help="Label source: the argmax of the trained Q-network, or the actions the agent "
        "actually took as recorded in the history.",
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

    # The trailing --epochs epochs, each [n_steps, n_agents, window_size, n_features].
    # Epochs are concatenated along the step axis, which history_reshape groups by
    # agent, so the row order and the agent ids stay the same as for a single epoch.
    n_epochs = history.observations.shape[0]
    keep = n_epochs if args.epochs == 0 else min(args.epochs, n_epochs)

    # A recorded observation is the state *after* its step: its newest window slot
    # already holds the action taken at that step. The action it leads to is the
    # next one, so pair observation t with action t+1 -- labelling it with action t
    # instead leaks the answer into the features and distills "repeat last action".
    # The pairing is per epoch: the last step of one epoch does not lead to the
    # first action of the next, so each epoch drops its own last step.
    observations = np.concatenate(
        [np.asarray(history.observations[e, :-1]) for e in range(n_epochs - keep, n_epochs)]
    )
    recorded = np.concatenate(
        [np.asarray(history.actions[e, 1:]) for e in range(n_epochs - keep, n_epochs)]
    )  # [keep * (n_steps - 1), n_agents]
    n_steps, _, window_size, _ = observations.shape
    print(f"Pooling {keep} of {n_epochs} epochs: {n_steps} steps per agent")

    # The features stay raw: ltc.run hands the distilled agents the raw observation
    # (use_raw_obs covers sr-jax and forester), so this is what they see at replay.
    # Build flat feature matrix: [n_agents * n_steps, window_size * n_features]
    # transpose [n_steps, n_agents, w, f] -> [n_agents, n_steps, w, f] then flatten last two dims
    XX = history_reshape(observations)
    if args.labels == "actions":
        # Agent-major, matching history_reshape's row order.
        actions = np.asarray(recorded).T.flatten()
    else:
        # The history records the raw observation, but the Q-network acted on the
        # normalized one (ltc.run feeds agents normalize_obs(obs), which rescales the
        # retransmission counter and the idle counter). Feeding it raw counts puts the
        # input an order of magnitude out of range: its greedy action then claims TX in
        # 73% of the steps where the agent actually transmitted in 8.5%. Normalized, the
        # two agree on 99.7%. Only the qvals labels need this, and it is the one pass
        # over the pooled epochs that costs real memory, so it stays out of the
        # --labels actions path.
        qvals = compute_qvals(
            params, state, normalize_obs(observations), jax.random.key(42)
        )  # [n_agents, n_steps, n_actions]
        actions = np.asarray(qvals.argmax(axis=-1)).flatten()  # [n_agents * n_steps]
    agent_ids = np.repeat(np.arange(n_agents), n_steps)

    df = pd.DataFrame(XX, columns=build_column_names(window_size))
    df["agent"] = agent_ids
    df["action"] = actions

    out_path = args.output if args.output else str(history_file).replace(".pkl.lz4", ".csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")
