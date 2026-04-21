import jax
import jax.numpy as jnp

from ltc.sim.constants import *
from ltc.sim.sim import is_buffer_empty
from ltc.utils.structs import ObsFeatureInputs, ObsFeatureConfig


def no_transmission(args):
    _, buffer_state, _, _, _, _ = args
    return jax.lax.cond(is_buffer_empty(buffer_state), idle_empty_buffer, idle_full_buffer, args)


def idle_empty_buffer(_):
    reward = EMPTY_BUFFER_REWARD
    ret_c = 0
    no_tx = 0
    return reward, ret_c, no_tx


def idle_full_buffer(args):
    _, _, _, _, no_tx, key = args
    rnd_factor = jax.random.normal(key) * SAFE_IDLE_PERIOD_STD
    rnd_factor = jnp.round(rnd_factor).astype(int)
    safe_period = SAFE_IDLE_PERIOD + rnd_factor
    args += (safe_period,)
    return jax.lax.cond(no_tx < safe_period, no_transmission_short, no_transmission_long, args)


def no_transmission_short(args):
    _, _, ret_c, _, no_tx, _, _ = args
    reward = NO_TX_REWARD
    no_tx = no_tx + 1
    return reward, ret_c, no_tx


def no_transmission_long(args):
    _, _, ret_c, _, no_tx, _, safe_period = args
    scale = jax.lax.min(1., (no_tx - safe_period + 1) / PENALIZED_IDLE_PERIOD)
    reward = scale * NO_TX_PENALTY
    no_tx = no_tx + 1
    return reward, ret_c, no_tx


def transmission(args):
    _, _, _, channel_state, _, _ = args
    return jax.lax.cond(channel_state == 1, transmission_without_collision, transmission_with_collision, args)


def transmission_with_collision(args):
    _, _, ret_c, _, _, _ = args
    return jax.lax.cond(ret_c < MAX_RETRANSMISSION, retransmission, max_retransmission_collision, args)


def max_retransmission_collision(_):
    reward = MAX_RETRANSMISSION_PENALTY
    ret_c = 0
    no_tx = 0
    return reward, ret_c, no_tx


def retransmission(args):
    _, _, ret_c, _, _, _ = args
    reward = COLLISION_PENALTY
    ret_c = ret_c + 1
    no_tx = 0
    return reward, ret_c, no_tx


def transmission_without_collision(args):
    _, buffer_state, _, _, _, _ = args
    return jax.lax.cond(~is_buffer_empty(buffer_state), successful_transmission, empty_buffer_transmission, args)


def empty_buffer_transmission(_):
    reward = EMPTY_TX_PENALTY
    ret_c = 0
    no_tx = 0
    return reward, ret_c, no_tx


def successful_transmission(args):
    _, _, ret_c, _, _, _ = args
    reward = TX_REWARD
    ret_c = 0
    no_tx = 0
    return reward, ret_c, no_tx


def _oldest_packet_age_norm(buffer_state, buffer_birth_state, current_step):
    valid = buffer_state != EMPTY_PACKET_ID
    oldest_birth = jnp.min(jnp.where(valid, buffer_birth_state, current_step))
    age = jnp.where(jnp.any(valid), current_step - oldest_birth, -1)
    queue_size = jnp.maximum(buffer_state.shape[0], 1)
    return jnp.where(age >= 0, age.astype(jnp.float32) / queue_size, -1.0)


def _build_observation_entry(
    buffer_state, new_buffer_state, new_buffer_birth_state, action, channel_state,
    current_step, obs_features, obs_config, ret_c, no_tx
):
    buffer_count = jnp.sum((new_buffer_state != EMPTY_PACKET_ID).astype(jnp.int32))
    buffer_occupancy_pct = buffer_count.astype(jnp.float32) / new_buffer_state.shape[0]
    oldest_packet_age_norm = _oldest_packet_age_norm(new_buffer_state, new_buffer_birth_state, current_step)

    cs_busy = jnp.where(action == Actions.CS.value, channel_state != 0, -1).astype(jnp.float32)
    cs_tx_same_type = jnp.where(
        obs_config.enable_cs_tx_same_type,
        jnp.where(jnp.logical_and(action == Actions.CS.value, channel_state == 1), obs_features.cs_tx_same_type_now, -1.0),
        -1.0,
    )
    tx_collision_other = jnp.where(
        obs_config.enable_tx_collision_other,
        jnp.where(jnp.logical_and(action == Actions.TX.value, channel_state == -1), obs_features.tx_collision_other_now, -1.0),
        -1.0,
    )

    return jnp.array([
        buffer_occupancy_pct,
        buffer_count.astype(jnp.float32),
        oldest_packet_age_norm,
        obs_features.traffic_mean_arrival_rate,
        cs_busy,
        cs_tx_same_type.astype(jnp.float32),
        tx_collision_other.astype(jnp.float32),
        obs_features.channel_occupancy_pct_window,
        obs_features.channel_collisions_pct_window,
        action.astype(jnp.float32),
        obs_features.back_pct,
        ret_c.astype(jnp.float32),
        no_tx.astype(jnp.float32),
        obs_features.unique_ltc_tx_window,
    ], dtype=jnp.float32)


def process_output_i(
    buffer_state,
    new_buffer_state,
    new_buffer_birth_state,
    power_state,
    channel_state,
    obs,
    action,
    terminal,
    key,
    current_step,
    obs_features,
    obs_config,
    roll_obs,
):
    ret_c = obs[-1, ObsIdx.STATUS_RETRY_COUNTER].astype(jnp.int32)
    no_tx = obs[-1, ObsIdx.STATUS_NO_TX_COUNTER].astype(jnp.int32)
    args = (action, buffer_state, ret_c, channel_state, no_tx, key)

    reward, ret_c, no_tx = jax.lax.cond(action == Actions.TX.value, transmission, no_transmission, args)
    reward = jnp.where(terminal, 0., reward)

    channel_state = jnp.where(action == Actions.CS.value, channel_state, -1)
    power = jnp.where(
        action == Actions.TX.value, power_state - TX_CONSUMPTION,
        jnp.where(
            action == Actions.CS.value, power_state - CS_CONSUMPTION,
            jnp.where(
                action == Actions.IDLE.value, power_state - IDLE_CONSUMPTION,
                power_state
            )
        )
    )

    obs_t = _build_observation_entry(
        buffer_state, new_buffer_state, new_buffer_birth_state, action, channel_state,
        current_step, obs_features, obs_config, ret_c, no_tx
    )
    obs = jax.lax.cond(
        roll_obs,
        lambda x: jnp.roll(x, -1, axis=0).at[-1].set(obs_t),
        lambda x: x.at[-1].set(obs_t),
        obs,
    )

    return obs, reward, power


def process_output(
    buffer_states,
    new_buffer_states,
    new_buffer_birth_states,
    power_states,
    channel_state,
    obs,
    actions,
    terminals,
    key,
    current_step,
    obs_features,
    obs_config,
    roll_obs=True,
):
    roll_obs = jnp.broadcast_to(jnp.asarray(roll_obs), actions.shape)
    return jax.vmap(process_output_i, in_axes=(0, 0, 0, 0, None, 0, 0, 0, None, None, 0, None, 0))(
        buffer_states,
        new_buffer_states,
        new_buffer_birth_states,
        power_states,
        channel_state,
        obs,
        actions,
        terminals,
        key,
        current_step,
        obs_features,
        obs_config,
        roll_obs,
    )
