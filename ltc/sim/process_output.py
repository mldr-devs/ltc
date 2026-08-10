from functools import partial

import jax
import jax.numpy as jnp

from ltc.sim.constants import *


def normalize_obs(obs):
    obs = obs.astype(jnp.float32)
    obs = obs.at[..., 2].divide(MAX_RETRANSMISSION)
    obs = obs.at[..., 3].divide(SAFE_IDLE_PERIOD + PENALIZED_IDLE_PERIOD)
    return obs


def no_transmission(args):
    _, buffer_state, _, _, _, _ = args
    return jax.lax.cond(buffer_state == 0, idle_empty_buffer, idle_full_buffer, args)


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
    return jax.lax.cond(buffer_state > 0, successful_transmission, empty_buffer_transmission, args)


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


def process_output_i(buffer_state, new_buffer_state, power_state, channel_state, obs, action, terminal, key,
                     observed_channel_state=None, perfect_channel_obs=False):
    _, _, ret_c, no_tx, _, _ = obs[-1]
    args = (action, buffer_state, ret_c, channel_state, no_tx, key)

    reward, ret_c, no_tx = jax.lax.cond(action == Actions.TX.value, transmission, no_transmission, args)
    reward = jnp.where(terminal, 0., reward)

    if observed_channel_state is not None:
        channel_state = observed_channel_state
    if not perfect_channel_obs:
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

    action_tx = (action == Actions.TX.value).astype(int)
    action_cs = (action == Actions.CS.value).astype(int)
    obs_t = jnp.array([new_buffer_state, channel_state, ret_c, no_tx, action_tx, action_cs])
    obs = jnp.roll(obs, -1, axis=0)
    obs = obs.at[-1].set(obs_t)

    return obs, reward, power


def process_output(buffer_states, new_buffer_states, power_states, channel_state, obs, actions, terminals, key,
                   observed_channel_states=None, perfect_channel_obs=False, reward_fn=None):
    step_fn = partial(process_output_i, perfect_channel_obs=perfect_channel_obs)
    observed_axis = None if observed_channel_states is None else 0
    obs, rewards, powers = jax.vmap(step_fn, in_axes=(0, 0, 0, None, 0, 0, 0, None, observed_axis))(
        buffer_states, new_buffer_states, power_states, channel_state, obs, actions, terminals, key,
        observed_channel_states
    )
    if reward_fn is not None:
        rewards = reward_fn(rewards, channel_state, actions, new_buffer_states, terminals)
    return obs, rewards, powers
