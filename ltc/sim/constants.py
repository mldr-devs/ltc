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
NO_TX_PENALTY = -1.0
EMPTY_TX_PENALTY = -0.5
COLLISION_PENALTY = -1.0
MAX_RETRANSMISSION_PENALTY = -1.0

"""
Tx slot duration (s)
https://ieeexplore.ieee.org/document/8930559
"""
TAU = 5.484 * 1e-3
