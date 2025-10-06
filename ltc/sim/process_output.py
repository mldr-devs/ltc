import jax
import jax.numpy as jnp

from ltc.sim.constants import *


def process_output_i(buffer_state, new_buffer_state, power_state, channel_state, obs, action, terminal):
    reward = channel_state == 1
    reward = jnp.where(terminal, 0., reward)

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

    obs_t = jnp.array([channel_state, action, new_buffer_state])
    obs = jnp.roll(obs, -1, axis=0)
    obs = obs.at[-1].set(obs_t)

    return obs, reward, power


def process_output(buffer_states, new_buffer_states, power_states, channel_state, obs, actions, terminals):
    channel_states = jnp.full(buffer_states.shape[0], channel_state)
    return jax.vmap(process_output_i)(buffer_states, new_buffer_states, power_states, channel_states, obs, actions, terminals)
