from enum import Enum


class Actions(Enum):
    """
    Actions for the agent
    """
    TX = 0
    CS = 1
    IDLE = 2


"""
Time scaling: one TX slot occupies TX_SLOTS mini-slots.
During one TX, CS agents can observe TX_SLOTS times.
"""
TX_SLOTS = 20

"""
Simulation parameters
"""
MAX_RETRANSMISSION = 8
SAFE_IDLE_FRAMES = 5
SAFE_IDLE_PERIOD_STD = 0
PENALIZED_IDLE_FRAMES = 25

TRAFFIC_LEVELS = {
    'low':    {'rho': 0.25, 'scale': 0.0, 'f3dB': 1.0},
    'mid':    {'rho': 0.50, 'scale': 0.0, 'f3dB': 1.0},
    'bursty': {'rho': 0.50, 'scale': 5.0, 'f3dB': 0.1},
}
SATURATED_LOC = 5.0

"""
Rewards and penalties
"""
TX_REWARD = 1.0
EMPTY_BUFFER_REWARD = 0.5
NO_TX_REWARD = 0.0
NO_TX_PENALTY = -1.0
COLLISION_PENALTY = -1.0

TX_SIZE_THRESHOLD = 3
TX_SIZE_PENALTY_WINDOW = 2
TX_SIZE_PENALTY = -1.0
TX_LTC_COLLISION_PENALTY = -1.0
TX_COEX_COLLISION_PENALTY = -5.0
TX_EMPTY_BUFFER_PENALTY = -0.5
TX_MAX_RETRANSMISSION_PENALTY = -1.0

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

"""
Queue representation
"""
EMPTY_PACKET_ID = -1

"""
Macro-action defaults
"""
MAX_MACRO_DURATION = 5
LEGACY_TX_DURATION = 5

"""
Observation normalization constants
"""
OBS_UNIQUE_LTC_TX_NORM = 10
OBS_TRAFFIC_LOG_RANGE = 5.0
OBS_DURATION_NORM = MAX_MACRO_DURATION


class ObsIdx:
    BUFFER_OCCUPANCY_PCT = 0
    BUFFER_OLDEST_AGE_NORM = 1
    TRAFFIC_MEAN_ARRIVAL_RATE = 2
    CHANNEL_LAST_CS_BUSY = 3
    CHANNEL_LAST_CS_TX_SAME_TYPE = 4
    CHANNEL_LAST_TX_COLLISION_OTHER = 5
    CHANNEL_SENSED_BUSY_PCT_WINDOW = 6
    ACTION_TX = 7
    ACTION_CS = 8
    ACTION_IDLE = 9
    ACTION_DURATION = 10
    ACTION_BACK_PCT = 11
    STATUS_RETRY_COUNTER = 12
    STATUS_NO_TX_COUNTER = 13
    STATUS_UNIQUE_LTC_TX_WINDOW = 14


OBS_SIZE = 15
