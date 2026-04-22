from dataclasses import dataclass
from typing import Any
import jax
from reinforced_lib.agents import AgentState


@jax.tree_util.register_dataclass
@dataclass
class MacroActionState:
    remaining: jax.Array
    action_types: jax.Array
    reward_accum: jax.Array
    tx_success_accum: jax.Array
    tx_collision_accum: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class ObsTrackerState:
    buffer_birth_steps: jax.Array
    arrival_hist: jax.Array
    planned_tx_hist: jax.Array
    success_tx_hist: jax.Array
    channel_busy_hist: jax.Array
    collision_hist: jax.Array
    staged_tx: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class ObsFeatureInputs:
    traffic_mean_arrival_rate: jax.Array
    channel_occupancy_pct_window: jax.Array
    channel_collisions_pct_window: jax.Array
    back_pct: jax.Array
    unique_ltc_tx_window: jax.Array
    cs_tx_same_type_now: jax.Array
    tx_collision_other_now: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class ObsFeatureConfig:
    enable_cs_tx_same_type: bool
    enable_tx_collision_other: bool


@jax.tree_util.register_dataclass
@dataclass
class ActionDecodingResults:
    drl_action_types: jax.Array
    drl_durations: jax.Array
    drl_staged_next: jax.Array
    legacy_action_types: jax.Array
    legacy_durations: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class Carry:
    drl_states: AgentState
    legacy_states: AgentState
    traffic_states: Any
    packet_seqs: jax.Array
    buffer_states: jax.Array
    tx_hist: jax.Array
    power_states: jax.Array
    channel_state: int
    key: jax.random.PRNGKey
    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminals: jax.Array
    active: jax.Array
    macro: MacroActionState
    obs_tracker: ObsTrackerState


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
    active: jax.Array
    weights_histogram: jax.Array
    weights_bin_edges: jax.Array
    successful_packets: jax.Array
