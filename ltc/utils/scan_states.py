from dataclasses import dataclass

import jax
from reinforced_lib.agents import AgentState

from ltc.sim import ModelState


@jax.tree_util.register_dataclass
@dataclass
class Carry:
    drl_states: AgentState
    dcf_states: AgentState
    traffic_states: ModelState
    buffer_states: jax.Array
    power_states: jax.Array
    channel_state: int
    key: jax.random.PRNGKey
    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminals: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class Output:
    dcf_states: AgentState
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminals: jax.Array
    buffer_states: jax.Array
    power_states: jax.Array
    new_frames: jax.Array
    channel_state: int
