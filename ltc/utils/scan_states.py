from dataclasses import dataclass

import jax
from reinforced_lib.agents import AgentState

from ltc.sim import ModelState


@jax.tree_util.register_dataclass
@dataclass
class Carry:
    drl_states: AgentState
    legacy_states: AgentState
    traffic_states: ModelState
    buffer_states: jax.Array
    power_states: jax.Array
    d2lt: jax.Array
    throughputs: jax.Array
    global_obs: jax.Array
    channel_state: int
    key: jax.random.PRNGKey
    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminals: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class Output:
    legacy_states: AgentState
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminals: jax.Array
    buffer_states: jax.Array
    power_states: jax.Array
    new_frames: jax.Array
    channel_state: int
    weights_histogram: jax.Array
    weights_bin_edges: jax.Array
