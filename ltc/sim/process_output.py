import jax
import jax.numpy as jnp

from ltc.sim.constants import *


def no_transmission(args):
    _, _, _, no_tx = args
    return jax.lax.cond(no_tx < SAFE_IDLE_PERIOD, no_transmission_short, no_transmission_long, args)


def no_transmission_short(args):
    buffer_state, ret_c, _, no_tx = args
    reward = NO_TX_REWARD
    no_tx = jax.lax.select(buffer_state == 0, 0, no_tx + 1)
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
    return jax.lax.cond(buffer_state > 0, successful_transmission, empty_buffer_transmission, args)


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


def process_output_i(buffer_state, new_buffer_state, power_state, channel_state, obs, action, terminal):
    _, _, ret_c, no_tx, _ = obs[-1]
    args = (buffer_state, ret_c, channel_state, no_tx)
    reward, ret_c, no_tx = jax.lax.cond(action == Actions.TX.value, transmission, no_transmission, args)

    reward = jnp.where(jax.lax.bitwise_or(terminal, action == Actions.IDLE.value), NO_TX_REWARD, reward)
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

    obs_t = jnp.array([new_buffer_state, channel_state, ret_c, no_tx, power])
    obs = jnp.roll(obs, -1, axis=0)
    obs = obs.at[-1].set(obs_t)

    return obs, reward, power


def process_output(buffer_states, new_buffer_states, power_states, channel_state, obs, actions, terminals):
    channel_states = jnp.full(buffer_states.shape[0], channel_state)
    return jax.vmap(process_output_i)(buffer_states, new_buffer_states, power_states, channel_states, obs, actions, terminals)
