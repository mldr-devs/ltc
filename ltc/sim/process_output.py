from functools import partial

import jax
import jax.numpy as jnp

from ltc.sim.constants import *
from ltc.sim.sim import is_buffer_empty


def no_transmission(args, tx_slots):
    _, buffer_state, _, _, _, _ = args
    return jax.lax.cond(
        is_buffer_empty(buffer_state), idle_empty_buffer, partial(idle_full_buffer, tx_slots=tx_slots), args
    )


def idle_empty_buffer(_):
    reward = EMPTY_BUFFER_REWARD
    ret_c = 0
    no_tx = 0
    return reward, ret_c, no_tx


def idle_full_buffer(args, tx_slots):
    _, _, _, _, no_tx, key = args
    rnd_factor = jax.random.normal(key) * SAFE_IDLE_PERIOD_STD
    rnd_factor = jnp.round(rnd_factor).astype(jnp.int32)
    safe_period = SAFE_IDLE_FRAMES * tx_slots + rnd_factor
    args += (safe_period,)
    return jax.lax.cond(
        no_tx < safe_period, no_transmission_short, partial(no_transmission_long, tx_slots=tx_slots), args
    )


def no_transmission_short(args):
    _, _, ret_c, _, no_tx, _, _ = args
    reward = NO_TX_REWARD
    no_tx = no_tx + 1
    return reward, ret_c, no_tx


def no_transmission_long(args, tx_slots):
    _, _, ret_c, _, no_tx, _, safe_period = args
    scale = jax.lax.min(1., (no_tx - safe_period + 1) / (PENALIZED_IDLE_FRAMES * tx_slots))
    reward = scale * NO_TX_PENALTY
    no_tx = no_tx + 1
    return reward, ret_c, no_tx


def reset_counters(_):
    ret_c = 0
    no_tx = 0
    return 0.0, ret_c, no_tx


def tx_macro_reward(
    k_tx, k_tx_planned, k_coll_ltc, k_coll_coex, tx_started_empty, retry_exhausted,
    header_collision, ack_collision, header_collision_other, ack_collision_other, tx_slots,
):
    hdr_ack = header_collision | ack_collision
    k_tx_eff = jnp.where(hdr_ack, 0.0, k_tx)

    # Header/ACK collision makes all frames unreadable: all planned sub-windows become collisions
    collision_with_other = (header_collision & header_collision_other) | (ack_collision & ack_collision_other)
    k_coll_ltc_eff = jnp.where(hdr_ack, jnp.where(~collision_with_other, k_tx_planned, 0.0), k_coll_ltc)
    k_coll_coex_eff = jnp.where(hdr_ack, jnp.where(collision_with_other, k_tx_planned, 0.0), k_coll_coex)

    k_coll = k_coll_ltc_eff + k_coll_coex_eff
    any_collision = k_coll > 0

    net_data_slots = jnp.maximum(k_tx_eff * tx_slots - 2.0, k_tx_eff)
    success_reward = jnp.where(
        k_tx_eff > 0,
        TX_REWARD * net_data_slots / tx_slots,
        0.0,
    )

    # Gated on a non-empty buffer (the note's `b > 0`): a TX on an empty buffer already pays
    # P_empty for every planned subframe and must not be charged for its length on top.
    size_penalty = jnp.where(
        ~tx_started_empty & (k_tx_planned >= TX_SIZE_THRESHOLD),
        TX_SIZE_PENALTY * jnp.minimum(1.0, (k_tx_planned - TX_SIZE_THRESHOLD + 1.0) / TX_SIZE_PENALTY_WINDOW),
        0.0,
    )

    # Exhausting the retransmission budget replaces the per-collision penalty, it does not add to it
    collision_reward = jnp.where(
        any_collision,
        jnp.where(
            retry_exhausted,
            TX_MAX_RETRANSMISSION_PENALTY,
            TX_LTC_COLLISION_PENALTY * k_coll_ltc_eff + TX_COEX_COLLISION_PENALTY * k_coll_coex_eff,
        ),
        0.0,
    )

    # Empty-buffer penalty scales with planned duration
    empty_penalty = jnp.where(
        jnp.logical_and(tx_started_empty, ~any_collision),
        TX_EMPTY_BUFFER_PENALTY * k_tx_planned,
        0.0,
    )

    return success_reward + size_penalty + collision_reward + empty_penalty


def oldest_packet_age_norm(buffer_state, buffer_birth_state, current_step):
    valid = buffer_state != EMPTY_PACKET_ID
    oldest_birth = jnp.min(jnp.where(valid, buffer_birth_state, current_step))
    age = jnp.where(jnp.any(valid), current_step - oldest_birth, -1)
    return jnp.where(age >= 0, age.astype(jnp.float32), -1.0)


def build_observation_entry(
    buffer_state, new_buffer_state, new_buffer_birth_state, action, action_duration, channel_state,
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

    action_tx   = jnp.where(action == Actions.TX.value,   1.0, 0.0)
    action_cs   = jnp.where(action == Actions.CS.value,   1.0, 0.0)
    action_idle = jnp.where(action == Actions.IDLE.value, 1.0, 0.0)

    return jnp.array([
        buffer_occupancy_pct,
        packet_age_norm,
        obs_features.traffic_mean_arrival_rate,
        cs_busy,
        cs_tx_same_type.astype(jnp.float32),
        tx_collision_other.astype(jnp.float32),
        obs_features.sensed_busy_pct_window,
        action_tx,
        action_cs,
        action_idle,
        action_duration.astype(jnp.float32),
        obs_features.back_pct,
        ret_c.astype(jnp.float32),
        no_tx.astype(jnp.float32),
        obs_features.unique_ltc_tx_window,
    ], dtype=jnp.float32)


def normalize_obs(obs, queue_size, tx_slots):
    raw_age = obs[..., ObsIdx.BUFFER_OLDEST_AGE_NORM]
    obs = obs.at[..., ObsIdx.BUFFER_OLDEST_AGE_NORM].set(
        jnp.where(raw_age >= 0, jnp.minimum(raw_age / (SAFE_IDLE_FRAMES * tx_slots), 1.0), raw_age)
    )
    raw_rate = obs[..., ObsIdx.TRAFFIC_MEAN_ARRIVAL_RATE]
    obs = obs.at[..., ObsIdx.TRAFFIC_MEAN_ARRIVAL_RATE].set(jnp.where(
        raw_rate < 0, raw_rate,
        jnp.clip((jnp.log(jnp.maximum(raw_rate, 1e-6)) + OBS_TRAFFIC_LOG_RANGE) / (2 * OBS_TRAFFIC_LOG_RANGE), 0.0, 1.0),
    ))
    obs = obs.at[..., ObsIdx.ACTION_DURATION].divide(OBS_DURATION_NORM)
    obs = obs.at[..., ObsIdx.STATUS_RETRY_COUNTER].divide(MAX_RETRANSMISSION)
    obs = obs.at[..., ObsIdx.STATUS_NO_TX_COUNTER].divide((SAFE_IDLE_FRAMES + PENALIZED_IDLE_FRAMES) * tx_slots)
    obs = obs.at[..., ObsIdx.STATUS_UNIQUE_LTC_TX_WINDOW].divide(OBS_UNIQUE_LTC_TX_NORM)
    return obs


def process_output_i(
    buffer_state,
    new_buffer_state,
    new_buffer_birth_state,
    power_state,
    channel_state,
    obs,
    action,
    action_duration,
    terminal,
    key,
    current_step,
    obs_features,
    obs_config,
    roll_obs,
    done_now,
    k_tx,
    k_tx_planned,
    k_coll_ltc,
    k_coll_coex,
    header_collision,
    header_collision_other,
    ack_collision,
    ack_collision_other,
    initial_ret_c,
    tx_started_empty,
    tx_macro_success,
    tx_macro_failed,
    retry_exhausted,
    tx_slots,
):
    old_ret_c = obs[-1, ObsIdx.STATUS_RETRY_COUNTER].astype(jnp.int64)
    no_tx     = obs[-1, ObsIdx.STATUS_NO_TX_COUNTER].astype(jnp.int64)

    is_tx = action == Actions.TX.value

    # CS/IDLE: get slot reward, no_tx update, and ret_c (resets if buffer empty)
    cs_args = (action, buffer_state, old_ret_c, channel_state, no_tx, key)
    cs_reward, cs_ret_c, cs_no_tx = no_transmission(cs_args, tx_slots)

    # TX: no_tx always resets (agent is transmitting); slot reward is 0 (macro reward below)
    no_tx       = jnp.where(is_tx, jnp.zeros_like(no_tx), cs_no_tx)
    slot_reward = jnp.where(is_tx, 0.0, cs_reward)

    # ret_c settles once per macro TX: a subframe success cannot count if the ACK later fails
    ret_c = jnp.where(is_tx, old_ret_c, cs_ret_c)          # CS uses cs logic; TX holds by default
    ret_c = jnp.where(tx_macro_success, 0, ret_c)
    ret_c = jnp.where(tx_macro_failed & ~retry_exhausted, initial_ret_c + 1, ret_c)
    ret_c = jnp.where(retry_exhausted, 0, ret_c)

    # The reward always uses the true collision type; obs_config only decides what the agent sees.
    macro_reward = tx_macro_reward(
        k_tx, k_tx_planned, k_coll_ltc, k_coll_coex, tx_started_empty, retry_exhausted,
        header_collision, ack_collision, header_collision_other, ack_collision_other, tx_slots,
    )
    reward = jnp.where(
        action == Actions.TX.value,
        jnp.where(done_now, macro_reward, 0.0),
        slot_reward,
    )
    reward = jnp.where(terminal, 0., reward)

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
        buffer_state, new_buffer_state, new_buffer_birth_state, action, action_duration, channel_state,
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
    action_durations=None,
    roll_obs=True,
    done_now=False,
    k_tx=None,
    k_tx_planned=None,
    k_coll_ltc=None,
    k_coll_coex=None,
    header_collision=None,
    header_collision_other=None,
    ack_collision=None,
    ack_collision_other=None,
    initial_ret_c=None,
    tx_started_empty=None,
    tx_macro_success=None,
    tx_macro_failed=None,
    retry_exhausted=None,
    tx_slots=TX_SLOTS,
):
    n = actions.shape[0]
    keys = jax.random.split(key, n)

    def bcast(x, default, dtype=None):
        x = default if x is None else x
        return jnp.broadcast_to(jnp.asarray(x, dtype=dtype) if dtype else jnp.asarray(x), (n,))

    action_durations = bcast(action_durations, 0, jnp.int32)
    roll_obs = bcast(roll_obs, False)
    done_now = bcast(done_now, False)
    k_tx = bcast(k_tx, 0.0, jnp.float32)
    k_tx_planned = bcast(k_tx_planned, 0.0, jnp.float32)
    k_coll_ltc = bcast(k_coll_ltc, 0.0, jnp.float32)
    k_coll_coex = bcast(k_coll_coex, 0.0, jnp.float32)
    header_collision = bcast(header_collision, False)
    header_collision_other = bcast(header_collision_other, False)
    ack_collision = bcast(ack_collision, False)
    ack_collision_other = bcast(ack_collision_other, False)
    initial_ret_c = bcast(initial_ret_c, 0, jnp.int32)
    tx_started_empty = bcast(tx_started_empty, False)
    tx_macro_success = bcast(tx_macro_success, False)
    tx_macro_failed = bcast(tx_macro_failed, False)
    retry_exhausted = bcast(retry_exhausted, False)

    return jax.vmap(
        process_output_i,
        in_axes=(0, 0, 0, 0, None, 0, 0, 0, 0, 0, None, 0, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None),
    )(
        buffer_states,
        new_buffer_states,
        new_buffer_birth_states,
        power_states,
        channel_state,
        obs,
        actions,
        action_durations,
        terminals,
        keys,
        current_step,
        obs_features,
        obs_config,
        roll_obs,
        done_now,
        k_tx,
        k_tx_planned,
        k_coll_ltc,
        k_coll_coex,
        header_collision,
        header_collision_other,
        ack_collision,
        ack_collision_other,
        initial_ret_c,
        tx_started_empty,
        tx_macro_success,
        tx_macro_failed,
        retry_exhausted,
        tx_slots,
    )
