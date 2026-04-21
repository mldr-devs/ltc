"""Organized state structures for the macro-action simulator.

This module defines nested dataclasses for logically grouping state fields
within the Carry object, improving readability and maintainability.
"""

from dataclasses import dataclass
import jax


@jax.tree_util.register_dataclass
@dataclass
class MacroActionState:
    """State for tracking macro-action execution per agent.
    
    Fields:
        remaining: Steps left in current macro-action for each agent.
        action_types: Primitive action type (TX/CS/IDLE) for current macro.
        reward_accum: Accumulated reward across macro-action slots.
        tx_success_accum: Count of successful TX subframes in current macro.
        tx_collision_accum: Count of collided TX subframes in current macro.
    """
    remaining: "jax.Array"
    action_types: "jax.Array"
    reward_accum: "jax.Array"
    tx_success_accum: "jax.Array"
    tx_collision_accum: "jax.Array"


@jax.tree_util.register_dataclass
@dataclass
class ObsTrackerState:
    """State for tracking observation metrics and packet histories.
    
    Fields:
        buffer_birth_steps: Arrival step for each packet in queue.
        arrival_hist: Rolling window of per-slot arrival counts.
        planned_tx_hist: Rolling window of planned TX attempts per agent.
        success_tx_hist: Rolling window of successful TX per agent.
        channel_busy_hist: Rolling window of channel busy readings.
        collision_hist: Rolling window of collision outcomes.
        staged_tx: Per-agent count of staged TX actions (for TX_STAGE/COMMIT).
    """
    buffer_birth_steps: "jax.Array"
    arrival_hist: "jax.Array"
    planned_tx_hist: "jax.Array"
    success_tx_hist: "jax.Array"
    channel_busy_hist: "jax.Array"
    collision_hist: "jax.Array"
    staged_tx: "jax.Array"
