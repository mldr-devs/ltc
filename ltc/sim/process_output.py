import jax
import jax.numpy as jnp

from ltc.sim.constants import *


def no_transmission(args):
    _, _, _, no_tx = args
    return jax.lax.cond(no_tx < SAFE_IDLE_PERIOD, no_transmission_short, no_transmission_long, args)


def no_transmission_short(args):
    _, ret_c, _, no_tx = args
    reward = NO_TX_REWARD
    no_tx = no_tx + 1
    return reward, ret_c, no_tx


def no_transmission_long(args):
    _, ret_c, _, no_tx = args
    scale = jax.lax.min(1., (no_tx - SAFE_IDLE_PERIOD + 1) / PENALIZED_IDLE_PERIOD)
    reward = scale * NO_TX_PENALTY
    no_tx = no_tx + 1
    return reward, ret_c, no_tx


def transmission(args):
    _, _, channel_state, _ = args
    return jax.lax.cond(channel_state == 1, transmission_without_collision, transmission_with_collision, args)


def transmission_with_collision(args):
    _, ret_c, _, _ = args
    return jax.lax.cond(ret_c < MAX_RETRANSMISSION, retransmission, max_retransmission_collision, args)


def max_retransmission_collision(_):
    reward = MAX_RETRANSMISSION_PENALTY
    ret_c = 0
    no_tx = 0
    return reward, ret_c, no_tx


def retransmission(args):
    _, ret_c, _, _ = args
    reward = COLLISION_PENALTY
    ret_c = ret_c + 1
    no_tx = 0
    return reward, ret_c, no_tx


def transmission_without_collision(args):
    buffer_state, _, _, _ = args
    return jax.lax.cond(buffer_state == 1, successful_transmission, empty_buffer_transmission, args)


def empty_buffer_transmission(_):
    reward = EMPTY_TX_PENALTY
    ret_c = 0
    no_tx = 0
    return reward, ret_c, no_tx


def successful_transmission(args):
    _, ret_c, _, _ = args
    reward = TX_REWARD / (ret_c + 1)
    ret_c = 0
    no_tx = 0
    return reward, ret_c, no_tx


def process_output_i(buffer_state, new_buffer_state, action, channel_state, obs):
    _, _, ret_c, no_tx = obs[-1]
    args = (buffer_state, ret_c, channel_state, no_tx)
    reward, ret_c, no_tx = jax.lax.cond(action == 1, transmission, no_transmission, args)

    channel_state = jnp.abs(channel_state)
    obs_t = jnp.array([new_buffer_state, channel_state, ret_c, no_tx])
    obs = jnp.roll(obs, -1, axis=0)
    obs = obs.at[-1].set(obs_t)

    return obs, reward


def process_output(buffer_states, new_buffer_states, actions, channel_state, obs):
    return jax.vmap(process_output_i, in_axes=(0, 0, 0, None, 0))(buffer_states, new_buffer_states, actions, channel_state, obs)
