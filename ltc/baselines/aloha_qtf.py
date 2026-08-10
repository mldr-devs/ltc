"""
Implementation of the ALOHA-QTF algorithm from:
"Approaching Fair Collision-Free Channel Access with Slotted ALOHA Using
Collaborative Policy-Based Reinforcement Learning" (de Alfaro et al., 2020)

ALOHA-QTF extends ALOHA-QT with fairness mechanisms:
- Tracks active nodes via sliding window of transmitter IDs
- Computes fair and requested bandwidth
- Scales weight updates based on fairness ratio
- Conditional slot relinquishment (only when using more than fair share)
"""

import jax
import jax.numpy as jnp
from chex import dataclass
from reinforced_lib.agents import BaseAgent, AgentState

from ltc.sim.constants import Actions, Features
from ltc.sim.features import select_features


# Pre-compute policy tree structure for efficiency
# For tree depth N=8, we have policies at levels 0..8
# Level m has 2^m policies, total = 2^0 + 2^1 + ... + 2^8 = 2^9 - 1 = 511
N = 8
NUM_POLICIES = (1 << (N + 1)) - 1  # 511 policies
WINDOW_SIZE = 1 << N  # 256

# Pre-compute level starts and periods for each policy index
# _POLICY_LEVELS[idx] = m (the level/depth of policy at index idx)
# _POLICY_OFFSETS[idx] = i (the offset within period for policy at index idx)
# _POLICY_PERIODS[idx] = 2^m (the period of policy at index idx)
def _build_policy_tables():
    """Build lookup tables for policy properties."""
    levels = []
    offsets = []
    periods = []
    for m in range(N + 1):
        period = 1 << m  # 2^m
        for i in range(period):
            levels.append(m)
            offsets.append(i)
            periods.append(period)
    return (
        jnp.array(levels, dtype=jnp.int32),
        jnp.array(offsets, dtype=jnp.int32),
        jnp.array(periods, dtype=jnp.int32),
    )

_POLICY_LEVELS, _POLICY_OFFSETS, _POLICY_PERIODS = _build_policy_tables()


def policy_to_index(i: int, m: int) -> int:
    """Convert policy (i, 2^m) to flat array index.

    Level 0: policy (0,1) -> index 0
    Level 1: policies (0,2), (1,2) -> indices 1, 2
    Level m: policies (i, 2^m) for i in [0, 2^m) -> indices [2^m - 1, 2^(m+1) - 1)
    """
    return (1 << m) - 1 + i  # 2^m - 1 + i


def get_enabled_mask(t: int) -> jax.Array:
    """Return boolean mask of policies enabled at time t.

    Policy (i, 2^m) is enabled if t mod 2^m == i.
    Uses vectorized operations for JAX efficiency.
    """
    # For each policy, check if t mod period == offset
    return (t % _POLICY_PERIODS) == _POLICY_OFFSETS


def estimate_n_hat(id_window: jax.Array, max_id_seen: int) -> jax.Array:
    """Estimate number of active nodes from ID window.

    N_hat = count of distinct IDs (excluding -1 placeholder) in window.
    Only counts IDs in range [0, max_id_seen].
    Truly distributed: no knowledge of total stations required.

    Args:
        id_window: Sliding window of station IDs, shape (WINDOW_SIZE,)
        max_id_seen: Maximum station ID observed so far

    Returns:
        int: Estimated number of active nodes (at least 1)
    """
    def check_id_present(station_id):
        # Only count valid IDs (0 to max_id_seen)
        return jnp.any(id_window == station_id) & (station_id <= max_id_seen)

    # Check presence of each possible station ID
    max_possible = 256  # Upper bound for vectorization
    station_ids = jnp.arange(max_possible)
    present = jax.vmap(check_id_present)(station_ids)
    n_hat = jnp.sum(present)

    return jnp.maximum(1, n_hat)


def compute_fair_bandwidth(n_hat: jax.Array) -> jax.Array:
    """Compute fair bandwidth share.

    b_f = 1 / max(1, N_hat)
    """
    return 1.0 / jnp.maximum(1.0, n_hat.astype(jnp.float32))


def compute_requested_bandwidth(weights: jax.Array, eta: float = 0.95) -> jax.Array:
    """Compute requested bandwidth from active policies.

    Active policies: max-weight policy OR weight > η (same as transmission decision).
    b_r = sum of 1/period for active policies, excluding descendants.

    From paper Section IV.B (Policy Selection):
    "A policy σ ∈ P is selected as active in a round if:
    1) either σ is the policy with the maximal weight
    2) or w_σ ≥ w_h, where w_h = 0.95"

    From paper Section V.A:
    "Remove from B every policy (i, m) such that there is (i', m') in B
    with m' < m and i mod m' = i'"
    """
    # Active set A_t: max-weight policy OR policies above threshold
    max_idx = jnp.argmax(weights)
    indices = jnp.arange(NUM_POLICIES)
    active = jnp.logical_or(weights > eta, indices == max_idx)

    # For each policy, check if it has an active ancestor
    # Policy at index idx has offset=_POLICY_OFFSETS[idx] and period=_POLICY_PERIODS[idx]
    # An ancestor (i', m') satisfies: m' < m and offset mod m' == i'

    def has_active_ancestor(idx):
        """Check if policy at idx has any active ancestor."""
        my_offset = _POLICY_OFFSETS[idx]
        my_period = _POLICY_PERIODS[idx]

        def check_ancestor(ancestor_idx):
            ancestor_period = _POLICY_PERIODS[ancestor_idx]
            ancestor_offset = _POLICY_OFFSETS[ancestor_idx]
            # Ancestor must have smaller period and my offset must match
            is_ancestor = (ancestor_period < my_period) & \
                         ((my_offset % ancestor_period) == ancestor_offset) & \
                         active[ancestor_idx]
            return is_ancestor

        # Check all potential ancestors
        ancestor_indices = jnp.arange(NUM_POLICIES)
        has_ancestor = jnp.any(jax.vmap(check_ancestor)(ancestor_indices))
        return has_ancestor

    # Mark policies that have active ancestors (these are descendants to exclude)
    is_descendant = jax.vmap(has_active_ancestor)(jnp.arange(NUM_POLICIES))

    # Only count active policies that are not descendants of other active policies
    count_policy = active & ~is_descendant
    bandwidth = jnp.sum(jnp.where(count_policy, 1.0 / _POLICY_PERIODS, 0.0))

    return jnp.clip(bandwidth, 0.0, 1.0)


def update_id_window(
    id_window: jax.Array,
    window_ptr: int,
    channel_state: jax.Array,
    successful_station_id: jax.Array,
    key: jax.random.PRNGKey,
    max_id_seen: int
) -> tuple:
    """Update the sliding ID window based on channel outcome.

    Args:
        id_window: Current window of IDs
        window_ptr: Current circular buffer position
        channel_state: 1=success, 0=empty, -1=collision
        successful_station_id: ID of successful transmitter (-1 if none)
        key: Random key for collision case
        max_id_seen: Maximum station ID observed so far

    Returns:
        Updated (id_window, window_ptr)

    Rules from paper:
        - Success (S): add successful_station_id
        - Empty (E): add -1 (placeholder)
        - Collision (C): add random ID from [0, max_id_seen + 1)
          This may over-estimate N_hat on collisions (paper's intent)
    """
    is_success = channel_state == 1
    is_empty = channel_state == 0

    # Random ID from [0, max_id_seen + 1) for collisions
    random_id = jax.random.randint(key, (), 0, jnp.maximum(1, max_id_seen + 1))

    new_id = jnp.where(
        is_success, successful_station_id,
        jnp.where(is_empty, -1, random_id)
    )

    new_window = id_window.at[window_ptr].set(new_id)
    new_ptr = (window_ptr + 1) % WINDOW_SIZE

    return new_window, new_ptr


@dataclass
class ALOHAQTFState(AgentState):
    """State for ALOHA-QTF agent with fairness.

    Attributes:
        weights: Policy weights, shape (511,). Higher weight = more likely to use.
        t: Time slot counter, incremented each step.
        id_window: Sliding window of node IDs for N_hat estimation, shape (WINDOW_SIZE,).
                   Contains station IDs for success, -1 for empty, random ID for collision.
        node_id: This node's unique identifier for fairness tracking.
        window_ptr: Current position in circular buffer (0 to WINDOW_SIZE-1).
        max_id_seen: Maximum station ID observed so far (for distributed operation).
    """
    weights: jax.Array      # Shape: (NUM_POLICIES,) = (511,)
    t: int                  # Time slot counter
    id_window: jax.Array    # Shape: (WINDOW_SIZE,) = (256,)
    node_id: int            # This agent's unique ID
    window_ptr: int         # Circular buffer pointer
    max_id_seen: int        # Maximum station ID seen so far


class ALOHAQTF(BaseAgent):
    """
    ALOHA-QTF: Fair slotted ALOHA with policy trees.

    This implementation includes fairness mechanisms from Section V of the paper:
    - Sliding window tracking of transmitter IDs
    - Fair and requested bandwidth estimation
    - Fairness-scaled weight updates
    - Conditional slot relinquishment
    """

    FEATURES = (Features.BUFFER, Features.CHANNEL)
    USES_AUX = True

    # Constants from paper
    N = 8
    NUM_POLICIES = (1 << (N + 1)) - 1  # 511
    WINDOW_SIZE = 1 << N  # 256

    W_INIT = 0.25
    ALPHA_PLUS = 0.2
    ALPHA_MINUS = -0.5
    GAMMA_0 = 0.1
    GAMMA_1 = 1.2
    ETA = 0.95
    EPSILON_R = 0.02

    def __init__(self):
        """Initialize and JIT compile all methods."""
        self.init = jax.jit(self.init)
        self.update = jax.jit(self.update)
        self.sample = jax.jit(self.sample)

    @staticmethod
    def init(key: jax.random.PRNGKey, node_id: int) -> ALOHAQTFState:
        """Initialize ALOHA-QTF state with fairness tracking.

        Weight initialization (from paper):
        w_(i,2^k) = W_INIT * (1 - GAMMA_0 + GAMMA_0 * X) / GAMMA_1^k

        Higher-bandwidth policies (smaller periods) get higher initial weights.

        Args:
            key: Random key for weight initialization
            node_id: Unique identifier for this node (0 to n-1)
        """
        key, weight_key = jax.random.split(key)

        noise = jax.random.uniform(weight_key, (ALOHAQTF.NUM_POLICIES,))
        level_scale = jnp.power(ALOHAQTF.GAMMA_1, -_POLICY_LEVELS.astype(jnp.float32))
        weights = ALOHAQTF.W_INIT * (1 - ALOHAQTF.GAMMA_0 + ALOHAQTF.GAMMA_0 * noise) * level_scale

        id_window = jnp.full(ALOHAQTF.WINDOW_SIZE, -1, dtype=jnp.int32)

        return ALOHAQTFState(
            weights=weights,
            t=0,
            id_window=id_window,
            node_id=node_id,
            window_ptr=0,
            max_id_seen=node_id,  # Start with own ID as maximum
        )

    @staticmethod
    def update(
        state: ALOHAQTFState,
        key: jax.random.PRNGKey,
        env_state: jax.Array,
        action: int,
        reward: float,
        terminal: bool,
        successful_station_id: int = -1,
        outcome: int = 0
    ) -> ALOHAQTFState:
        """Update policy weights with fairness-based scaling.

        Key differences from ALOHA-QT:
        1. Maintain sliding window of transmitter IDs
        2. Estimate N_hat (active nodes) from window
        3. Compute fair (b_f) and requested (b_r) bandwidth
        4. Scale weight updates by fairness ratio
        5. Conditional relinquishment: only if b_r > b_f

        Args:
            state: Current agent state
            key: Random key for stochastic updates
            env_state: Observation array, shape (window_size, 5)
                       env_state[-1] = [buffer, channel, ret_c, no_tx, action_tx, action_cs]
            action: Action taken (TX=0, CS=1, IDLE=2)
            reward: Reward received
            terminal: Whether episode terminated
            successful_station_id: ID of station that transmitted successfully (-1 if none)

        Returns:
            Updated agent state
        """
        key, update_key, relinquish_key, redist_key, window_key = jax.random.split(key, 5)

        _, channel_state = select_features(env_state, ALOHAQTF.FEATURES)[-1]

        # === FAIRNESS: Update max_id_seen when we observe a successful transmission ===
        new_max_id_seen = jnp.where(
            (channel_state == 1) & (successful_station_id >= 0),
            jnp.maximum(state.max_id_seen, successful_station_id),
            state.max_id_seen
        )

        # === FAIRNESS: Update ID window ===
        new_id_window, new_window_ptr = update_id_window(
            state.id_window, state.window_ptr, channel_state,
            successful_station_id, window_key, new_max_id_seen
        )

        # === FAIRNESS: Compute bandwidth metrics ===
        n_hat = estimate_n_hat(new_id_window, new_max_id_seen)
        b_f = compute_fair_bandwidth(n_hat)
        b_r = compute_requested_bandwidth(state.weights, ALOHAQTF.ETA)

        ratio = b_r / jnp.maximum(b_f, 1e-6)

        # === Determine outcome ===
        # From paper Algorithm 1 and Section IV.D:
        # - Positive (α = α⁺ = 0.2): (W, E) or (T, S) - promote policies
        # - Negative (α = α⁻ = -0.5): ALL other cases: (W, S), (W, C), (T, C)
        #
        # When waiting and observing success/collision, the enabled policies
        # would have caused a collision if we had transmitted, so they must
        # be demoted to learn to avoid those slots.
        did_tx = action == Actions.TX.value
        is_success = channel_state == 1
        is_empty = channel_state == 0

        positive_outcome = jnp.logical_or(
            jnp.logical_and(~did_tx, is_empty),    # (W, E)
            jnp.logical_and(did_tx, is_success)    # (T, S)
        )
        # All other outcomes are negative: (W, S), (W, C), (T, C)

        # === FAIRNESS: Scaled alpha values ===
        # From paper Section V.B equations (2) and (3):
        # - Demotion (alpha < 0): scale by min(1, sqrt(b_r/b_f))
        # - Promotion (alpha > 0): scale by max(0, 1 - (b_r/b_f)^2)
        demotion_scale = jnp.minimum(1.0, jnp.sqrt(ratio))
        promotion_scale = jnp.maximum(0.0, 1.0 - ratio ** 2)

        # Paper: α⁺ for positive outcomes, α⁻ for ALL other cases
        alpha_base = jnp.where(
            positive_outcome, ALOHAQTF.ALPHA_PLUS, ALOHAQTF.ALPHA_MINUS
        )

        # Apply fairness scaling per paper equations (2) and (3)
        alpha_scaled = jnp.where(
            alpha_base > 0,
            alpha_base * promotion_scale,
            alpha_base * demotion_scale
        )

        # === Weight update ===
        enabled = get_enabled_mask(state.t)
        rand = jax.random.uniform(update_key, (ALOHAQTF.NUM_POLICIES,))
        multiplier = jnp.exp(rand * alpha_scaled)
        new_weights = jnp.where(enabled, state.weights * multiplier, state.weights)

        # === FAIRNESS: Conditional relinquishment ===
        # Only relinquish if b_r > b_f (using more than fair share)
        should_relinquish_condition = b_r > b_f
        should_relinquish_random = jax.random.uniform(relinquish_key) < ALOHAQTF.EPSILON_R
        should_relinquish = jnp.logical_and(should_relinquish_condition, should_relinquish_random)

        new_weights = jnp.where(
            jnp.logical_and(should_relinquish, enabled),
            0.0,
            new_weights
        )

        # === Weight redistribution ===
        w_before = jnp.sum(state.weights)
        w_after = jnp.sum(new_weights)
        delta = w_before - w_after

        w_init_total = ALOHAQTF.W_INIT * ALOHAQTF.NUM_POLICIES
        should_redistribute = jnp.logical_and(delta > 0, w_after < w_init_total)

        redist_rand = jax.random.uniform(redist_key, (ALOHAQTF.NUM_POLICIES,))
        redist_normalized = redist_rand / jnp.sum(redist_rand)

        new_weights = jnp.where(
            should_redistribute,
            new_weights + delta * redist_normalized,
            new_weights
        )

        new_weights = jnp.clip(new_weights, 0.0, 1.0)
        new_t = state.t + 1

        return ALOHAQTFState(
            weights=new_weights,
            t=new_t,
            id_window=new_id_window,
            node_id=state.node_id,
            window_ptr=new_window_ptr,
            max_id_seen=new_max_id_seen,
        )

    @staticmethod
    def sample(
        state: ALOHAQTFState,
        key: jax.random.PRNGKey,
        env_state: jax.Array
    ) -> int:
        """Select action based on active policies.

        Args:
            state: Current agent state
            key: Random key (unused in deterministic selection)
            env_state: Observation array, shape (window_size, 5)
                       env_state[-1] = [buffer, channel, ret_c, no_tx, action_tx, action_cs]

        Returns:
            Action: TX (0) if should transmit, IDLE (2) otherwise
        """
        buffer, _ = select_features(env_state, ALOHAQTF.FEATURES)[-1]
        has_data = buffer > 0

        max_idx = jnp.argmax(state.weights)
        indices = jnp.arange(ALOHAQTF.NUM_POLICIES)
        active = jnp.logical_or(
            state.weights > ALOHAQTF.ETA,
            indices == max_idx
        )

        enabled = get_enabled_mask(state.t)

        should_tx = jnp.logical_and(
            jnp.any(jnp.logical_and(active, enabled)),
            has_data
        )

        return jnp.where(should_tx, Actions.TX.value, Actions.CS.value)
