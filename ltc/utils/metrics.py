"""Throughput and fairness of an ``ltc.run`` rollout, from its history alone.

Numpy only, and deliberately free of any ``ltc.sim`` import: the package's
``__init__`` pulls in jax and the agents, and both callers here -- the comparison
plots and the Pareto-front selection -- want to read a history without paying for
that. ``Actions.TX`` is 0; it is a parameter rather than an import for the same
reason.
"""

import numpy as np

TX_ACTION = 0


def flatten(history):
    """Return ``(actions, buffer_before, channel)`` as ``(T, n_agents)`` / ``(T,)``.

    A history is recorded per epoch, so ``actions`` arrives as
    ``[n_epochs, n_steps, n_agents]``; the epochs are concatenated into one
    timeline. ``buffer_before`` is the buffer as it stood *going into* the step,
    which is what decides whether a transmission had anything to carry.
    """
    actions = np.asarray(history.actions)
    buffer = np.asarray(history.buffer_states)
    channel = np.asarray(history.channel_state)

    if actions.ndim == 3:
        n_agents = actions.shape[2]
        actions = actions.reshape(-1, n_agents)
        buffer = buffer.reshape(-1, n_agents)
        channel = channel.reshape(-1)
    elif actions.ndim == 1:
        actions = actions.reshape(-1, 1)
        buffer = buffer.reshape(-1, 1)

    buffer_before = np.zeros_like(buffer)
    buffer_before[1:] = buffer[:-1]
    return actions, buffer_before, channel


def per_agent_success(history, tx_action: int = TX_ACTION):
    """Per-step, per-agent successful-transmission mask, shape ``(T, n_agents)``.

    A step counts only when the station transmitted, the channel came back clean
    (``channel_state == 1``, i.e. exactly one transmitter) and it actually had a
    frame buffered.
    """
    actions, buffer_before, channel = flatten(history)
    return (actions == tx_action) & (channel[:, None] == 1) & (buffer_before == 1)


def steady_state_metrics(history, last_percent: float = 0.1, tx_action: int = TX_ACTION):
    """Aggregate throughput and Jain's fairness over the final ``last_percent``.

    Returns ``(throughput, fairness)``. Throughput is frames per step summed over
    stations; fairness is 0 when nothing got through at all, which is the signature
    of a policy that has locked every station into a permanent collision.
    """
    success = per_agent_success(history, tx_action)
    total_steps, n_agents = success.shape
    start = int(total_steps * (1 - last_percent))

    tail = success[start:]
    agg_throughput = tail.sum(axis=1).mean()

    agent_throughput = tail.mean(axis=0)
    denom = n_agents * (agent_throughput**2).sum()
    fairness = 0.0 if denom == 0 else (agent_throughput.sum() ** 2) / denom
    return float(agg_throughput), float(fairness)
