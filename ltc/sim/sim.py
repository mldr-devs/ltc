import jax.numpy as jnp
import jax.random as jr

from ltc.sim.constants import Actions


def channel_state_selector(actions):
    """
    Returns a state of the channel due to how many STA transmit in the same time.
    :param actions: vector of stations that transmit at a given time
    :return:
    int: -1 if more than one STA transmit, 1 if exactly one STA transmit, 0 if noone transmit at the moment.
    """

    ones_count = jnp.sum(actions == Actions.TX.value)
    return jnp.where(
        ones_count > 1, -1,
        jnp.where(ones_count == 1, 1, 0)
    )


def buffer_clearing(buffer_states, actions):
    """
    Updates the buffer_states vector based on the action vector.
    Sets the value to 0 in buffer_states where there is a 1 in both vectors.
    :param buffer_states: vector with binary value of STAs' buffer occupation.
        0 - STA's buffer is empty
        1 - STA's buffer is full
    :param actions: binary vector describes STAs' actions
        0 - channel sensing
        1 - transmission
    :return:
        jnp.ndarray: updated buffer_states.
    """
    return jnp.where((buffer_states == 1) & (actions == Actions.TX.value), 0, buffer_states)


def add_new_frames(buffer_states, new_frames):
    """
    Updates the buffer_states by adding new frames from generator.
    :param buffer_states: vector with binary value of STAs' buffer occupation.
    :param new_frames: vector with new frames generated for each STA.
    :return:
        jnp.ndarray: updated buffer_states.
    """
    new_frames = (new_frames > 0).astype(int)
    return jnp.bitwise_or(buffer_states, new_frames)


def phy_interference(error_probability, sim_key):
    return jr.bernoulli(sim_key, error_probability).astype(int)


def simulate(buffer_states, new_frames, actions, sim_key, error_probability=0.05):
    channel_state = channel_state_selector(actions)
    phy_error = phy_interference(error_probability, sim_key)
    channel_state = jnp.where(phy_error == 1, -1, channel_state)
    buffer_states = jnp.where(channel_state == 1, buffer_clearing(buffer_states, actions), buffer_states)
    buffer_states = add_new_frames(buffer_states, new_frames)
    return buffer_states, channel_state, phy_error
