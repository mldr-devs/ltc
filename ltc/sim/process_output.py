import jax
import jax.numpy as jnp

from ltc.sim.constants import *


def no_transmission(args):
    action, buffer_state, _, _, no_tx = args
    return jax.lax.cond(
        action == Actions.IDLE.value,
        lambda: jax.lax.cond(buffer_state == 0, idle_empty_buffer, idle_full_buffer, args),
        lambda: jax.lax.cond(no_tx < SAFE_IDLE_PERIOD, no_transmission_short, no_transmission_long, args),
    )


def idle_empty_buffer(_):
    reward = EMPTY_BUFFER_REWARD
    ret_c = 0
    no_tx = 0
    return reward, ret_c, no_tx


def idle_full_buffer(args):
    _, _, ret_c, _, no_tx = args
    reward = NO_TX_REWARD
    no_tx = no_tx + 1
    return reward, ret_c, no_tx


def no_transmission_short(args):
    _, buffer_state, ret_c, _, no_tx = args
    reward = NO_TX_REWARD
    no_tx = jnp.where(buffer_state == 0, 0, no_tx + 1)
    return reward, ret_c, no_tx


def no_transmission_long(args):
    _, _, ret_c, _, no_tx = args
    scale = jax.lax.min(1., (no_tx - SAFE_IDLE_PERIOD + 1) / PENALIZED_IDLE_PERIOD)
    reward = scale * NO_TX_PENALTY
    no_tx = no_tx + 1
    return reward, ret_c, no_tx


def transmission(args):
    _, _, _, channel_state, _ = args
    return jax.lax.cond(channel_state == 1, transmission_without_collision, transmission_with_collision, args)


def transmission_with_collision(args):
    _, _, ret_c, _, _ = args
    return jax.lax.cond(ret_c < MAX_RETRANSMISSION, retransmission, max_retransmission_collision, args)


def max_retransmission_collision(_):
    reward = MAX_RETRANSMISSION_PENALTY
    ret_c = 0
    no_tx = 0
    return reward, ret_c, no_tx


def retransmission(args):
    _, _, ret_c, _, _ = args
    reward = COLLISION_PENALTY
    ret_c = ret_c + 1
    no_tx = 0
    return reward, ret_c, no_tx


def transmission_without_collision(args):
    _, buffer_state, _, _, _ = args
    return jax.lax.cond(buffer_state > 0, successful_transmission, empty_buffer_transmission, args)


def empty_buffer_transmission(_):
    reward = EMPTY_TX_PENALTY
    ret_c = 0
    no_tx = 0
    return reward, ret_c, no_tx


def successful_transmission(args):
    _, _, ret_c, _, _ = args
    reward = TX_REWARD / (ret_c + 1)
    ret_c = 0
    no_tx = 0
    return reward, ret_c, no_tx


def process_output_i(buffer_state, new_buffer_state, power_state, d2lt, idx, channel_state, obs, action, terminal):
    _, _, no_tx, _, _, ret_c, _ = obs[-1]
    args = (action, buffer_state, ret_c, channel_state, no_tx)

    d2lt_i = d2lt[idx]
    d2lt_mi = d2lt.at[idx].set(jnp.inf).min()
    d2lt_i, d2lt_mi = d2lt_i / (d2lt_i + d2lt_mi), d2lt_mi / (d2lt_i + d2lt_mi)

    reward, ret_c, no_tx = jax.lax.cond(action == Actions.TX.value, transmission, no_transmission, args)
    reward = jnp.where(terminal, 0., reward)

    channel_state = jnp.where(action == Actions.TX.value, channel_state == 1, jnp.abs(channel_state))
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

    obs_t = jnp.array([action, channel_state, 1, d2lt_i, d2lt_mi, ret_c, new_buffer_state])
    obs = jnp.roll(obs, -1, axis=0)
    obs = obs.at[-1].set(obs_t)

    return obs, reward, power


def process_output(buffer_states, new_buffer_states, power_states, d2lt, channel_state, obs, actions, terminals):
    idxs = jnp.arange(buffer_states.shape[0])
    return jax.vmap(process_output_i, in_axes=(0, 0, 0, None, 0, None, 0, 0, 0))(
        buffer_states, new_buffer_states, power_states, d2lt, idxs, channel_state, obs, actions, terminals
    )
