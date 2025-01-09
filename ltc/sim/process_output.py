import jax
import jax.numpy as jnp

from ltc.sim.constants import *


def no_transmission(args):
    buffer_state, r, channel_state = args
    reward = 0.0
    return reward, buffer_state, r


def transmission(args):
    buffer_state, r, channel_state = args
    return jax.lax.cond(channel_state == 0, transmission_without_collision, transmission_with_collision, args)


def transmission_with_collision(args):
    buffer_state, r, channel_state = args
    return jax.lax.cond(r < MAX_RETRANSMISSION, retransmission, max_retransmission_collision, args)


def max_retransmission_collision(args):
    buffer_state, r, channel_state = args
    buffer_state = 0
    r = 0
    reward = COLLISION_PENALTY
    return reward, buffer_state, r


def retransmission(args):
    buffer_state, r, channel_state = args
    reward = COLLISION_PENALTY
    buffer_state = buffer_state
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


@jax.jit
def process_rl_output(buffer_states, actions, channel_state, obs_i_t_minus, i):
    # update buffer state
    r = obs_i_t_minus[-1][2]
    channel_state = jnp.where(channel_state == -1, 1, channel_state)
    action = actions[i]
    buffer_state = buffer_states[i]
    args = (buffer_state, r, channel_state)
    R_i, buffer_state, r = jax.lax.cond(action == 1, transmission, no_transmission, args)
    buffer_states = buffer_states.at[i].set(buffer_state)

    # update history
    obs_t = jnp.array([buffer_states[i], channel_state, r])
    obs_i_t = jnp.roll(obs_i_t_minus, -1, axis=0)
    obs_i_t = obs_i_t.at[-1].set(obs_t)

    return buffer_states, obs_i_t, R_i
