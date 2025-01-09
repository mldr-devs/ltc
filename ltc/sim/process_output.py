import jax
import jax.numpy as jnp

from ltc.sim.constants import *


def no_transmission(args):
    buffer_state, r, channel_state = args
    reward = 0.0
    return reward, buffer_state, r


def transmission(args):
    buffer_state, r, channel_state = args
    return jax.lax.cond(channel_state == 1, transmission_without_collision, transmission_with_collision, args)


def transmission_with_collision(args):
    buffer_state, r, channel_state = args
    return jax.lax.cond(r < MAX_RETRANSMISSION, retransmission, max_retransmission_collision, args)


def max_retransmission_collision(_):
    buffer_state = 0
    r = 0
    reward = COLLISION_PENALTY
    return reward, buffer_state, r


def retransmission(args):
    buffer_state, r, channel_state = args
    reward = COLLISION_PENALTY
    r = r + 1
    return reward, buffer_state, r


def transmission_without_collision(args):
    buffer_state, r, channel_state = args
    return jax.lax.cond(buffer_state == 1, successful_transmission, empty_buffer_transmission, args)


def empty_buffer_transmission(args):
    buffer_state, r, channel_state = args
    reward = 0.0
    return reward, buffer_state, r


def successful_transmission(args):
    buffer_state, r, channel_state = args
    reward = (TX_REWARD / (r + 1) ** 2)
    buffer_state = 0
    r = 0
    return reward, buffer_state, r


def process_output_i(buffer_state, action, channel_state, obs):
    # update buffer state
    r = obs[-1, 2]
    args = (buffer_state, r, channel_state)
    R_i, buffer_state, r = jax.lax.cond(action == 1, transmission, no_transmission, args)

    # update history
    channel_state = jnp.abs(channel_state)
    obs_t = jnp.array([buffer_state, channel_state, r])
    obs = jnp.roll(obs, -1, axis=0)
    obs = obs.at[-1].set(obs_t)

    return buffer_state, obs, R_i


def process_output(buffer_states, actions, channel_state, obs):
    return jax.vmap(process_output_i, in_axes=(0, 0, None, 0))(buffer_states, actions, channel_state, obs)
