from enum import Enum
import jax.numpy as jnp


class Actions(Enum):
    """
    Actions for the agent
    """
    TX = 0
    CS = 1
    IDLE = 2
    ACK = 3


"""
Simulation parameters
"""
MAX_RETRANSMISSION = 8
SAFE_IDLE_PERIOD = 32
PENALIZED_IDLE_PERIOD = 32

"""
Rewards and penalties
"""
# TX_REWARD = 1.0
# ACK_REWARD = 1.21371
# EMPTY_BUFFER_REWARD = 0.5
# NO_TX_REWARD = 0.0
# NO_TX_PENALTY = -1.0
# EMPTY_TX_PENALTY = -0.5
# EMPTY_ACK_PENALTY = -0.2137
# COLLISION_PENALTY = -1.0
# MAX_RETRANSMISSION_PENALTY = -1.0

TX_REWARD = 1.0
ACK_REWARD = 1.0
EMPTY_BUFFER_REWARD = 0.5
NO_TX_REWARD = 0.0
NO_TX_PENALTY = -1.0
EMPTY_TX_PENALTY = -0.5
EMPTY_ACK_PENALTY = -0.5
COLLISION_PENALTY = -1.0
MAX_RETRANSMISSION_PENALTY = -1.0

"""
Power consumption
"""
INITIAL_CAPACITY = 10000
TX_CONSUMPTION = 5
ACK_CONSUMPTION = 2
CS_CONSUMPTION = 1
IDLE_CONSUMPTION = 0

"""
Tx slot duration (s)
https://ieeexplore.ieee.org/document/8930559
"""
TAU = 5.484 * 1e-3


class Ack_state(Enum):
    """
    ACK state
    """
    SENT = 1
    NOT_SENT = 0


class Observation_indexes(Enum):
    """
    OBSERVATION INDEX
    """
    BUFFER_INDEX = 0
    CHANNEL_INDEX = 1
    RET_C_INDEX = 2
    NO_TX_INDEX = 3
    POWER_INDEX = 4
    ID_INDEX = 5


class Transmision_indexes(Enum):
    """
    TRANSMISSION INDEX
    """
    DESTINATION_INDEX = 0
    ACK_INDEX = 1


WINDOW_SIZE = 4
TRANSMISSION_HISTORY = jnp.zeros((WINDOW_SIZE+1, 2), dtype=int)
