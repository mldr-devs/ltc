from enum import Enum


class Actions(Enum):
    """
    Actions for the agent
    """
    TX = 0
    IDLE = 1


"""
Simulation parameters
"""
MAX_RETRANSMISSION = 8
SAFE_IDLE_PERIOD = 32
PENALIZED_IDLE_PERIOD = 32

"""
Rewards and penalties
"""
TX_REWARD = 1.0
NO_TX_REWARD = 0.0

"""
Power consumption
"""
INITIAL_CAPACITY = 10000
TX_CONSUMPTION = 5
IDLE_CONSUMPTION = 0

"""
Tx slot duration (s)
https://ieeexplore.ieee.org/document/8930559
"""
TAU = 5.484 * 1e-3
