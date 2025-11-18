import jax
import jax.numpy as jnp

from ltc.sim.constants import *


def process_output_i(buffer_state, new_buffer_state, power_state, d2lt, idx, channel_state, obs, action, terminal):
    *_, ret_c, _ = obs[-1]
    ret_c = jnp.where(action == Actions.TX.value, jnp.where(channel_state == 1, 0, ret_c + 1), ret_c)
    
    d2lt_i = d2lt[idx]
    d2lt_mi = d2lt.at[idx].set(jnp.inf).min()
    denominator = jnp.maximum(d2lt_i + d2lt_mi, 1)
    d2lt_i, d2lt_mi = d2lt_i / denominator, d2lt_mi / denominator

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

    obs_t = jnp.array([action, channel_state, 1, d2lt_i, d2lt_mi, d2lt[idx], ret_c, new_buffer_state])
    obs = jnp.roll(obs, -1, axis=0)
    obs = obs.at[-1].set(obs_t)

    return obs, power


def process_output(buffer_states, new_buffer_states, power_states, d2lt, throughputs, channel_state, obs, actions, terminals):
    tx_action = (actions == Actions.TX.value).astype(int)
    thr_t = (tx_action & (channel_state == 1)).astype(int)
    throughputs = jnp.roll(throughputs, -1, axis=1)
    throughputs = throughputs.at[:, -1].set(thr_t)

    priority = new_buffer_states / jnp.maximum(throughputs.mean(axis=1), 1)
    opt_action = (priority == priority.max()).astype(int)
    rewards_ind = 2 * (tx_action == opt_action).astype(float) - 1
    rewards_ind = jnp.where(terminals, 0., rewards_ind)

    reward_tot = jnp.where(
        channel_state == -1, COLLISION_PENALTY,
        jnp.where(
            channel_state == 0, NO_TX_REWARD,
            jnp.where(
                jnp.argmax(tx_action) == jnp.argmax(d2lt), TX_REWARD,
                d2lt[jnp.argmax(tx_action)] / jnp.maximum(d2lt.sum(), 1)
            )
        )
    )
    rewards = jnp.concatenate([jnp.array([reward_tot]), rewards_ind])

    idxs = jnp.arange(buffer_states.shape[0])
    obs, powers = jax.vmap(process_output_i, in_axes=(0, 0, 0, None, 0, None, 0, 0, 0))(
        buffer_states, new_buffer_states, power_states, d2lt, idxs, channel_state, obs, actions, terminals
    )

    return obs, rewards, throughputs, powers
