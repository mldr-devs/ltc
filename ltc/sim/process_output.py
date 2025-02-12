import jax
import jax.numpy as jnp

from ltc.sim.constants import *


def no_transmission(args):
    _, _, _, noTx = args
    return jax.lax.cond(noTx < MAX_NO_TX_WITHOUT_PENALTY, no_transmission_short, no_transmission_long, args)

def no_transmission_short(args):
    _, r, _, noTx = args
    reward = SHORT_NO_TX_REWARD
    noTx = noTx + 1
    return reward, r, noTx

def no_transmission_long(args):
    _, _, _, noTx = args
    return jax.lax.cond(noTx < MAX_NO_TX_WITH_PENALTY, no_transmission_penalty, max_no_transmission, args)

def max_no_transmission(args):
    _, r, _, noTx = args
    reward = MAX_NO_TX_REWARD
    noTx = 0
    return reward, r, noTx
def no_transmission_penalty(args):
    _, _, _, noTx = args
    return jax.lax.cond(noTx <= NO_TX_WITHOUT_MAX_PENALTY, no_transmission_scale_penalty, no_transmission_max_penalty, args)

def no_transmission_scale_penalty(args):
    _, r, _, noTx = args
    scale_index = (noTx - MAX_NO_TX_WITHOUT_PENALTY) / (NO_TX_WITHOUT_MAX_PENALTY - MAX_NO_TX_WITHOUT_PENALTY)
    reward = LONG_NO_TX_REWARD * scale_index
    noTx = noTx + 1
    return reward, r, noTx

def no_transmission_max_penalty(args):
    _, r, _, noTx = args
    reward = LONG_NO_TX_REWARD
    noTx = noTx + 1
    return reward, r, noTx

def transmission(args):
    _, _, channel_state, _ = args
    return jax.lax.cond(channel_state == 1, transmission_without_collision, transmission_with_collision, args)


def transmission_with_collision(args):
    _, r, _, _ = args
    return jax.lax.cond(r < MAX_RETRANSMISSION, retransmission, max_retransmission_collision, args)


def max_retransmission_collision(_):
    reward = MAX_RETRANSMISSION_PENALTY
    r = 0
    noTx = 0
    return reward, r, noTx


def retransmission(args):
    _, r, _, _ = args
    reward = COLLISION_PENALTY
    r = r + 1
    noTx = 0
    return reward, r, noTx


def transmission_without_collision(args):
    buffer_state, _, _, _ = args
    return jax.lax.cond(buffer_state == 1, successful_transmission, empty_buffer_transmission, args)


def empty_buffer_transmission(args):
    _, r, _, _ = args
    reward = EMPTY_TX_PENALTY
    noTx = 0
    return reward, r, noTx


def successful_transmission(args):
    _, r, _, _ = args
    reward = (TX_REWARD / (r + 1) ** 2)
    r = 0
    noTx = 0
    return reward, r, noTx


def process_output_i(buffer_state, new_buffer_state, action, channel_state, obs):
    _, _, r, noTx = obs[-1]
    args = (buffer_state, r, channel_state, noTx)
    reward, r, noTx = jax.lax.cond(action == 1, transmission, no_transmission, args)

    channel_state = jnp.abs(channel_state)
    obs_t = jnp.array([new_buffer_state, channel_state, r, noTx])
    obs = jnp.roll(obs, -1, axis=0)
    obs = obs.at[-1].set(obs_t)

    return obs, reward


def process_output(buffer_states, new_buffer_states, actions, channel_state, obs):
    return jax.vmap(process_output_i, in_axes=(0, 0, 0, None, 0))(buffer_states, new_buffer_states, actions, channel_state, obs)
