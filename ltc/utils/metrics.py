"""Throughput and fairness of an ``ltc.run`` rollout, from its history alone.

Numpy only, and deliberately free of any ``ltc.sim`` import: the package's
``__init__`` pulls in jax and the agents, and both callers here -- the comparison
plots and the Pareto-front selection -- want to read a history without paying for
that. ``Actions.TX`` is 0; it is a parameter rather than an import for the same
reason.
"""

import numpy as np

TX_ACTION = 0
# ltc.sim.sim.channel_state_selector: exactly one transmitter in the slot.
CHANNEL_SUCCESS = 1


def buffer_before(buffer):
    """The buffer as it stood *going into* each step, same shape as ``buffer``.

    Accepts the recorded ``[n_epochs, n_steps, n_agents]`` layout or an already
    flat ``[T, n_agents]`` timeline. The shift deliberately crosses epoch
    boundaries: ltc.run threads one ``Carry`` through every epoch's ``lax.scan``
    over a continuous ``global_steps``, so the epochs are consecutive chunks of a
    single rollout, not independent episodes. Only the very first step of the run
    has no predecessor.
    """
    flat = buffer.reshape(-1, buffer.shape[-1])
    shifted = np.zeros_like(flat)
    shifted[1:] = flat[:-1]
    return shifted.reshape(buffer.shape)


def success_mask(actions, buffer, channel, live=None, tx_action: int = TX_ACTION):
    """Successful-transmission mask, preserving whatever shape it is given.

    Three conditions, and the third is the one that is easy to forget: the station
    transmitted, the channel came back clean, *and* it had a frame to send. The
    simulator lets a station transmit on an empty buffer -- that is what
    ``EMPTY_TX_PENALTY`` exists for -- and such a slot occupies the medium and
    reads as ``CHANNEL_SUCCESS`` while carrying nothing. Counting it inflates
    throughput by 30% on the bursty teacher, where buffers are usually empty; under
    saturated traffic the two definitions agree to 0.04%.

    ``live`` is the optional presence mask for runs where stations join or leave.
    ``channel`` is per-slot, so it is broadcast over the station axis.
    """
    mask = (
        (actions == tx_action)
        & (channel[..., None] == CHANNEL_SUCCESS)
        & (buffer_before(buffer) == 1)
    )
    return mask if live is None else mask & live


def as_timeline(history):
    """Return ``(actions, buffer_before, channel)`` as ``(T, n_agents)`` / ``(T,)``.

    A history is recorded per epoch, so ``actions`` arrives as
    ``[n_epochs, n_steps, n_agents]``; the epochs are concatenated into one
    timeline. ``buffer_before`` is the buffer as it stood *going into* the step,
    which is what decides whether a transmission had anything to carry.

    Not to be confused with ``ltc.symbolic.util.history_reshape``, which collapses
    a different pair of axes for a different purpose: it turns the observation
    tensor ``[n_steps, n_agents, window_size, n_features]`` into the agent-major
    design matrix ``[n_agents * n_steps, window_size * n_features]`` that the
    distillation fits on. This one drops the epoch axis of the recorded rollout
    and derives a lagged buffer; there is no window or feature axis in sight.
    """
    actions, buffer, channel = _flat_arrays(history)
    return actions, buffer_before(buffer), channel


def _flat_arrays(history):
    """``(actions, buffer, channel)`` collapsed to one timeline, buffer unshifted."""
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

    return actions, buffer, channel


def per_agent_success(history, tx_action: int = TX_ACTION):
    """Per-step, per-agent successful-transmission mask, shape ``(T, n_agents)``."""
    return success_mask(*_flat_arrays(history), tx_action=tx_action)


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
