import jax
import jax.numpy as jnp

from ltc.sim.constants import *


def no_transmission(args):
    action, buffer_state, _, _, no_tx = args
    return jax.lax.cond(
        action == Actions.IDLE.value,
        lambda: jax.lax.cond(buffer_state == 0, zero_counters, increment_no_tx, args),
        lambda: jax.lax.cond(no_tx < SAFE_IDLE_PERIOD, no_transmission_short, increment_no_tx, args),
    )


def no_transmission_short(args):
    _, buffer_state, ret_c, _, no_tx = args
    no_tx = jnp.where(buffer_state == 0, 0, no_tx + 1)
    return ret_c, no_tx


def increment_no_tx(args):
    _, _, ret_c, _, no_tx = args
    no_tx = no_tx + 1
    return ret_c, no_tx


def transmission(args):
    _, _, _, channel_state, _ = args
    return jax.lax.cond(channel_state == 1, zero_counters, transmission_with_collision, args)


def transmission_with_collision(args):
    _, _, ret_c, _, _ = args
    return jax.lax.cond(ret_c < MAX_RETRANSMISSION, retransmission, zero_counters, args)


def retransmission(args):
    _, _, ret_c, _, _ = args
    ret_c = ret_c + 1
    no_tx = 0
    return ret_c, no_tx


def zero_counters(_):
    ret_c = 0
    no_tx = 0
    return ret_c, no_tx


def process_output_i(buffer_state, new_buffer_state, power_state, d2lt, idx, channel_state, obs, action, terminal):
    _, _, no_tx, _, _, ret_c, _ = obs[-1]
    args = (action, buffer_state, ret_c, channel_state, no_tx)

    d2lt_i = d2lt[idx]
    d2lt_mi = d2lt.at[idx].set(jnp.inf).min()
    d2lt_i, d2lt_mi = d2lt_i / (d2lt_i + d2lt_mi), d2lt_mi / (d2lt_i + d2lt_mi)

    ret_c, no_tx = jax.lax.cond(action == Actions.TX.value, transmission, no_transmission, args)

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

    return obs, power


def process_output(buffer_states, new_buffer_states, power_states, d2lt, throughputs, channel_state, obs, actions, terminals):
    global_obs = jnp.concatenate([actions, d2lt / d2lt.sum()])

    thr_t = ((actions == Actions.TX.value) & (channel_state == 1)).astype(int)
    throughputs = jnp.roll(throughputs, -1, axis=1)
    throughputs = throughputs.at[:, -1].set(thr_t)

    priority = new_buffer_states / (throughputs.mean(axis=1) + 1e-6)
    opt_action = (priority == priority.max()).astype(int)
    rewards_ind = 2 * (actions == opt_action).astype(float) - 1
    rewards_ind = jnp.where(terminals, 0., rewards_ind)

    reward_tot = jnp.where(
        channel_state == -1, COLLISION_PENALTY,
        jnp.where(
            channel_state == 0, NO_TX_REWARD,
            jnp.where(
                jnp.argmax(actions == Actions.TX.value) == jnp.argmax(d2lt), TX_REWARD,
                d2lt[jnp.argmax(actions == Actions.TX.value)] / d2lt.sum()
            )
        )
    )
    rewards = jnp.concatenate([rewards_ind, jnp.array([reward_tot])])

    idxs = jnp.arange(buffer_states.shape[0])
    obs, powers = jax.vmap(process_output_i, in_axes=(0, 0, 0, None, 0, None, 0, 0, 0))(
        buffer_states, new_buffer_states, power_states, d2lt, idxs, channel_state, obs, actions, terminals
    )

    return obs, rewards, throughputs, global_obs, powers
