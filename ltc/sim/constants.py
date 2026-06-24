from enum import Enum

import jax.numpy as jnp


class Actions(Enum):
    """
    Actions for the agent
    """
    TX = 0
    CS = 1
    IDLE = 2


"""
Simulation parameters
"""
MAX_RETRANSMISSION = 8
SAFE_IDLE_PERIOD = 25
SAFE_IDLE_PERIOD_STD = 3
PENALIZED_IDLE_PERIOD = 25

"""
Rewards and penalties
"""
TX_REWARD = 1.0
EMPTY_BUFFER_REWARD = 0.5
NO_TX_REWARD = 0.0
NO_TX_PENALTY = -1.0
EMPTY_TX_PENALTY = -0.5
COLLISION_PENALTY = -1.0
MAX_RETRANSMISSION_PENALTY = -1.0

def all_rewards() -> jnp.ndarray:
    k = jnp.arange(1, PENALIZED_IDLE_PERIOD + 1, dtype=jnp.float32)
    scaled_penalties = jnp.minimum(1.0, k / PENALIZED_IDLE_PERIOD) * NO_TX_PENALTY
    fixed = jnp.array([TX_REWARD, EMPTY_BUFFER_REWARD, NO_TX_REWARD, EMPTY_TX_PENALTY, COLLISION_PENALTY, MAX_RETRANSMISSION_PENALTY], dtype=jnp.float32)
    return jnp.unique(jnp.concatenate([scaled_penalties, fixed]))


"""
Power consumption
"""
INITIAL_CAPACITY = int(1e9)
TX_CONSUMPTION = 5
CS_CONSUMPTION = 1
IDLE_CONSUMPTION = 0

"""
Tx slot duration (s)
https://ieeexplore.ieee.org/document/8930559
"""
TAU = 5.484 * 1e-3
