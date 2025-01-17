import jax
import jax.numpy as jnp

from ltc.sim.constants import *


def no_transmission(args):
    _, r, _ = args
    reward = NO_TX_REWARD
    return reward, r


def transmission(args):
    _, _, channel_state = args
    return jax.lax.cond(channel_state == 1, transmission_without_collision, transmission_with_collision, args)


def transmission_with_collision(args):
    _, r, _ = args
    return jax.lax.cond(r < MAX_RETRANSMISSION, retransmission, max_retransmission_collision, args)


def max_retransmission_collision(_):
    reward = MAX_RETRANSMISSION_PENALTY
    r = 0
    return reward, r


def retransmission(args):
    _, r, _ = args
    reward = COLLISION_PENALTY
    r = r + 1
    return reward, r


def transmission_without_collision(args):
    buffer_state, _, _ = args
    return jax.lax.cond(buffer_state == 1, successful_transmission, empty_buffer_transmission, args)


def empty_buffer_transmission(args):
    _, r, _ = args
    reward = EMPTY_TX_PENALTY
    return reward, r


def successful_transmission(args):
    _, r, _ = args
    reward = (TX_REWARD / (r + 1) ** 2)
    r = 0
    return reward, r


def process_output_i(buffer_state, new_buffer_state, action, channel_state, obs):
    _, _, r = obs[-1]
    args = (buffer_state, r, channel_state)
    reward, r = jax.lax.cond(action == 1, transmission, no_transmission, args)

    channel_state = jnp.abs(channel_state)
    obs_t = jnp.array([new_buffer_state, channel_state, r])
    obs = jnp.roll(obs, -1, axis=0)
    obs = obs.at[-1].set(obs_t)

    return obs, reward


def process_output(buffer_states, new_buffer_states, actions, channel_state, obs):
    return jax.vmap(process_output_i, in_axes=(0, 0, 0, None, 0))(buffer_states, new_buffer_states, actions, channel_state, obs)
