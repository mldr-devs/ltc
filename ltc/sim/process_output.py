import jax
import jax.numpy as jnp

from ltc.sim.constants import *
from ltc.sim.sim import is_buffer_empty


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
    return jax.lax.cond(channel_state == 1, reset_counters, transmission_with_collision, args)


def transmission_with_collision(args):
    _, _, ret_c, _, _, _ = args
    return jax.lax.cond(ret_c < MAX_RETRANSMISSION, retransmission, reset_counters, args)


def reset_counters(_):
    ret_c = 0
    no_tx = 0
    return 0.0, ret_c, no_tx


def retransmission(args):
    _, _, ret_c, _, _, _ = args
    ret_c = ret_c + 1
    no_tx = 0
    return 0.0, ret_c, no_tx


def tx_macro_reward(
    k_tx, k_coll_ltc, k_coll_coex, tx_started_empty, initial_ret_c,
    header_collision, ack_collision, header_collision_other, ack_collision_other,
):
    hdr_ack = header_collision | ack_collision
    k_tx_eff = jnp.where(hdr_ack, 0.0, k_tx)

    # When header/ACK collides, attribute all k_tx frames as collisions of the same type
    collision_with_other = (header_collision & header_collision_other) | (ack_collision & ack_collision_other)
    k_coll_ltc_eff = k_coll_ltc + jnp.where(hdr_ack & ~collision_with_other, k_tx, 0.0)
    k_coll_coex_eff = k_coll_coex + jnp.where(hdr_ack & collision_with_other, k_tx, 0.0)

    k_coll = k_coll_ltc_eff + k_coll_coex_eff
    any_collision = k_coll > 0

    success_reward = jnp.where(
        k_tx_eff > 0,
        TX_REWARD * (k_tx_eff * TX_SLOTS - 2.0) / TX_SLOTS,
        0.0,
    )

    size_penalty = jnp.where(
        k_tx_eff >= TX_SIZE_THRESHOLD,
        TX_SIZE_PENALTY * jnp.minimum(1.0, (k_tx_eff - TX_SIZE_THRESHOLD + 1.0) / TX_SIZE_PENALTY_WINDOW),
        0.0,
    )

    collision_reward = jnp.where(
        any_collision,
        jnp.where(
            initial_ret_c >= MAX_RETRANSMISSION,
            TX_MAX_RETRANSMISSION_PENALTY,
            TX_LTC_COLLISION_PENALTY * k_coll_ltc_eff + TX_COEX_COLLISION_PENALTY * k_coll_coex_eff,
        ),
        0.0,
    )

    empty_penalty = jnp.where(
        jnp.logical_and(tx_started_empty, ~any_collision),
        TX_EMPTY_BUFFER_PENALTY,
        0.0,
    )

    return success_reward + size_penalty + collision_reward + empty_penalty


def oldest_packet_age_norm(buffer_state, buffer_birth_state, current_step):
    valid = buffer_state != EMPTY_PACKET_ID
    oldest_birth = jnp.min(jnp.where(valid, buffer_birth_state, current_step))
    age = jnp.where(jnp.any(valid), current_step - oldest_birth, -1)
    queue_size = jnp.maximum(buffer_state.shape[0], 1)
    return jnp.where(age >= 0, age.astype(jnp.float32) / queue_size, -1.0)


def build_observation_entry(
    buffer_state, new_buffer_state, new_buffer_birth_state, action, channel_state,
    current_step, obs_features, obs_config, ret_c, no_tx
):
    buffer_count = jnp.sum((new_buffer_state != EMPTY_PACKET_ID).astype(jnp.int32))
    buffer_occupancy_pct = buffer_count.astype(jnp.float32) / new_buffer_state.shape[0]
    packet_age_norm = oldest_packet_age_norm(new_buffer_state, new_buffer_birth_state, current_step)

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
        packet_age_norm,
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
    done_now,
    k_tx,
    k_coll_ltc,
    k_coll_coex,
    header_collision,
    header_collision_other,
    ack_collision,
    ack_collision_other,
    initial_ret_c,
    tx_started_empty,
):
    ret_c = obs[-1, ObsIdx.STATUS_RETRY_COUNTER].astype(jnp.int32)
    no_tx = obs[-1, ObsIdx.STATUS_NO_TX_COUNTER].astype(jnp.int32)
    args = (action, buffer_state, ret_c, channel_state, no_tx, key)

    # Per-slot call: updates ret_c/no_tx for observation; reward only used for CS/IDLE
    slot_reward, ret_c, no_tx = jax.lax.cond(action == Actions.TX.value, transmission, no_transmission, args)

    macro_reward = tx_macro_reward(
        k_tx, k_coll_ltc, k_coll_coex, tx_started_empty, initial_ret_c,
        header_collision, ack_collision, header_collision_other, ack_collision_other,
    )
    reward = jnp.where(
        action == Actions.TX.value,
        jnp.where(done_now, macro_reward, 0.0),
        slot_reward,
    )
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

    obs_t = build_observation_entry(
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
    done_now=False,
    k_tx=None,
    k_coll_ltc=None,
    k_coll_coex=None,
    header_collision=None,
    header_collision_other=None,
    ack_collision=None,
    ack_collision_other=None,
    initial_ret_c=None,
    tx_started_empty=None,
):
    n = actions.shape[0]
    roll_obs = jnp.broadcast_to(jnp.asarray(roll_obs), actions.shape)
    done_now = jnp.broadcast_to(jnp.asarray(done_now), actions.shape)
    k_tx = jnp.broadcast_to(jnp.asarray(k_tx if k_tx is not None else 0.0, dtype=jnp.float32), (n,))
    k_coll_ltc = jnp.broadcast_to(jnp.asarray(k_coll_ltc if k_coll_ltc is not None else 0.0, dtype=jnp.float32), (n,))
    k_coll_coex = jnp.broadcast_to(jnp.asarray(k_coll_coex if k_coll_coex is not None else 0.0, dtype=jnp.float32), (n,))
    header_collision = jnp.broadcast_to(jnp.asarray(header_collision if header_collision is not None else False), (n,))
    header_collision_other = jnp.broadcast_to(jnp.asarray(header_collision_other if header_collision_other is not None else False), (n,))
    ack_collision = jnp.broadcast_to(jnp.asarray(ack_collision if ack_collision is not None else False), (n,))
    ack_collision_other = jnp.broadcast_to(jnp.asarray(ack_collision_other if ack_collision_other is not None else False), (n,))
    initial_ret_c = jnp.broadcast_to(jnp.asarray(initial_ret_c if initial_ret_c is not None else 0, dtype=jnp.int32), (n,))
    tx_started_empty = jnp.broadcast_to(jnp.asarray(tx_started_empty if tx_started_empty is not None else False), (n,))

    return jax.vmap(process_output_i, in_axes=(0, 0, 0, 0, None, 0, 0, 0, None, None, 0, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
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
        done_now,
        k_tx,
        k_coll_ltc,
        k_coll_coex,
        header_collision,
        header_collision_other,
        ack_collision,
        ack_collision_other,
        initial_ret_c,
        tx_started_empty,
    )
