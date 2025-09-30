import jax
import jax.numpy as jnp

from ltc.sim.constants import *


def no_transmission(args):
    _, _, ret_c, _, _ = args
    reward = NO_TX_REWARD
    ret_c = 0
    no_tx = 0
    return reward, ret_c, no_tx

def transmission(args):
    _, _, _, channel_state, _ = args
    return jax.lax.cond(channel_state == 1, successful_transmission, transmission_with_collision, args)


def transmission_with_collision(args):
    _, _, ret_c, _, _ = args
    reward = NO_TX_REWARD
    ret_c = 0
    no_tx = 0
    return reward, ret_c, no_tx

def successful_transmission(args):
    _, _, ret_c, _, _ = args
    reward = TX_REWARD
    ret_c = 0
    no_tx = 0
    return reward, ret_c, no_tx


def process_output_i(buffer_state, new_buffer_state, power_state, channel_state, obs, action, terminal):
    ret_c = 0
    no_tx = 0
    args = (action, buffer_state, ret_c, channel_state, no_tx)

    reward, ret_c, no_tx = jax.lax.cond(action == Actions.TX.value, transmission, no_transmission, args)
    reward = jnp.where(terminal, 0., reward)

    # channel_state = jnp.where(action == Actions.CS.value, channel_state, -1)

    power = jnp.where(
        action == Actions.TX.value, power_state - TX_CONSUMPTION,
        power_state
    )

    obs_t = jnp.array([channel_state, action])
    obs = jnp.roll(obs, -1, axis=0)
    obs = obs.at[-1].set(obs_t)

    return obs, reward, power


def process_output(buffer_states, new_buffer_states, power_states, channel_state, obs, actions, terminals):
    channel_states = jnp.full(buffer_states.shape[0], channel_state)
    return jax.vmap(process_output_i)(buffer_states, new_buffer_states, power_states, channel_states, obs, actions, terminals)
