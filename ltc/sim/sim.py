import jax.numpy as jnp
import jax

from ltc.sim.constants import *


def channel_state_selector(actions):
    """
    Returns a state of the channel due to how many STA transmit in the same time.
    :param actions: vector of stations that transmit at a given time
    :return:
    int: -1 if more than one STA transmit, 1 if exactly one STA transmit, 0 if noone transmit at the moment.
    """

    transmitting = (actions == Actions.TX.value) | (actions == Actions.ACK.value)
    active_count = jnp.sum(transmitting)

    return jnp.where(
        active_count > 1, -1,
        jnp.where(active_count == 1, 1, 0)
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


def modify_transmission_history(actions, transmission_history, address):
    args = (actions, transmission_history, address)
    return jax.lax.cond(
        jnp.any(actions == Actions.ACK.value),
        ack_transmission,
        lambda a: jax.lax.cond(
            jnp.any(actions == Actions.TX.value),
            data_transmission,
            no_transmission,
            operand=a
        ),
        operand=args
    )


def data_transmission(args):
    actions, transmission_history, address = args
    transmission_history_t = jnp.array([address, AckState.NOT_SENT.value])
    transmission_history = jnp.roll(transmission_history, -1, axis=0)
    transmission_history = transmission_history.at[-1].set(transmission_history_t)
    return transmission_history


def ack_transmission(args):
    actions, transmission_history, address = args
    sta_num = jnp.argmax(actions == Actions.ACK.value)
    mask = (transmission_history[:, TransmissionIndex.DESTINATION.value] == sta_num) & (
            transmission_history[:, TransmissionIndex.ACK.value] == AckState.NOT_SENT.value)
    matching_rows = jnp.nonzero(mask, size=mask.shape[0], fill_value=-1)[0]
    transmission_history = transmission_history.at[matching_rows[0], TransmissionIndex.ACK.value].set(
        AckState.SENT.value)
    return transmission_history


def no_transmission(args):
    actions, transmission_history, address = args
    return transmission_history


def simulate(key, buffer_states, new_frames, actions, transmission_history, nWifi):
    channel_state = channel_state_selector(actions)
    address = jnp.where(channel_state == 1, jax.random.randint(key, shape=(), minval=0, maxval=nWifi), -1)
    transmission_history = jnp.where(channel_state == 1,
                                     modify_transmission_history(actions, transmission_history, address),
                                     transmission_history)
    buffer_states = jnp.where(channel_state == 1, buffer_clearing(buffer_states, actions), buffer_states)
    buffer_states = add_new_frames(buffer_states, new_frames)

    return buffer_states, channel_state, transmission_history
