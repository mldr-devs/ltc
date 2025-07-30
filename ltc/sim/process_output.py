import jax
import jax.numpy as jnp

from ltc.sim.constants import *

# obs = action, buffer_state, ret_c, channel_state, no_tx
def no_transmission(args):
    action, buffer_state, _, _, no_tx, _ = args
    return jax.lax.cond(
        action == Actions.IDLE.value,
        lambda: jax.lax.cond(buffer_state == 0, idle_empty_buffer, idle_full_buffer, args),
        lambda: jax.lax.cond(no_tx < SAFE_IDLE_PERIOD, no_transmission_short, no_transmission_long, args),
    )


def idle_empty_buffer(_):
    reward = EMPTY_BUFFER_REWARD
    ret_c = 0
    no_tx = 0
    is_ack = 0
    return reward, ret_c, no_tx, is_ack


def idle_full_buffer(args):
    _, _, ret_c, _, no_tx, _ = args
    reward = NO_TX_REWARD
    no_tx = no_tx + 1
    is_ack = 0
    return reward, ret_c, no_tx, is_ack


def no_transmission_short(args):
    _, buffer_state, ret_c, _, no_tx, _ = args
    reward = NO_TX_REWARD
    no_tx = jnp.where(buffer_state == 0, 0, no_tx + 1)
    is_ack = 0
    return reward, ret_c, no_tx, is_ack


def no_transmission_long(args):
    _, _, ret_c, _, no_tx, _ = args
    scale = jax.lax.min(1., (no_tx - SAFE_IDLE_PERIOD + 1) / PENALIZED_IDLE_PERIOD)
    reward = scale * NO_TX_PENALTY
    no_tx = no_tx + 1
    is_ack = 0
    return reward, ret_c, no_tx, is_ack


def transmission(args):
    _, _, _, channel_state, _, _ = args
    return jax.lax.cond(channel_state == 1, transmission_without_collision, transmission_with_collision, args)


def transmission_with_collision(args):
    _, _, ret_c, _, _, _ = args
    return jax.lax.cond(ret_c < MAX_RETRANSMISSION, retransmission, max_retransmission_collision, args)


def max_retransmission_collision(_):
    reward = MAX_RETRANSMISSION_PENALTY
    ret_c = 0
    no_tx = 0
    is_ack = 0
    return reward, ret_c, no_tx, is_ack


def retransmission(args):
    _, _, ret_c, _, _, _ = args
    reward = COLLISION_PENALTY
    ret_c = ret_c + 1
    no_tx = 0
    is_ack = 0
    return reward, ret_c, no_tx, is_ack


def transmission_without_collision(args):
    _, buffer_state, _, _, _, _ = args
    return jax.lax.cond(buffer_state > 0, successful_transmission, empty_buffer_transmission, args)


def empty_buffer_transmission(_):
    reward = EMPTY_TX_PENALTY
    ret_c = 0
    no_tx = 0
    is_ack = 0
    return reward, ret_c, no_tx, is_ack


def successful_transmission(args):
    _, _, ret_c, _, _, _ = args
    reward = TX_REWARD / (ret_c + 1)
    ret_c = 0
    no_tx = 0
    is_ack = 0
    return reward, ret_c, no_tx, is_ack

def transmission_ack(args):
    _, _, _, channel_state, _, _ = args
    return jax.lax.cond(channel_state == 1, transmission_ack_without_collision, transmission_with_collision, args)

def transmission_ack_without_collision(args):
    _, _, _, _, _, ack_flag = args
    return jax.lax.cond(ack_flag == 1, successful_ack_transmission, empty_ack_transmission, args)

def successful_ack_transmission(args):
    _, _, ret_c, _, _, _ = args
    reward = ACK_REWARD / (ret_c + 1)
    ret_c = 0
    no_tx = 0
    is_ack = 1
    return reward, ret_c, no_tx, is_ack

def empty_ack_transmission(_):
    reward = EMPTY_ACK_PENALTY
    ret_c = 0
    no_tx = 0
    is_ack = 0
    return reward, ret_c, no_tx, is_ack

def ack_sended(obs, dest_address, matching_rows):
    obs = obs.at[matching_rows[-1], ACK_INDEX].set(Ack_state.SENT)
    dest_address = obs[matching_rows[-1], SOURCE_ADDRESS_INDEX]

    return obs, dest_address

def no_ack_sended(obs, dest_address, matching_rows):
    obs = obs
    dest_address = dest_address
    return obs, dest_address

def process_output_i(buffer_state, new_buffer_state, power_state, channel_state, obs, action, terminal, dest_address, my_id):
    '''
    type - type of the obserwation 0-other 1-data 2-ack
    dest_address
    ack - is ack was sended for that obserwation [0/1]
    obs = buffer_state, channel_state, ret_c, no_tx, power, type, dest_address, ack
    '''
    _, _, ret_c, no_tx, _, _, _, _ = obs[-1]

    mask = (obs[:, DEST_ADDRESS_INDEX] == my_id) & (obs[:, ACK_INDEX] == Ack_state.NOT_SENT) & (obs[:, TYPE_INDEX] == OBSERVATION_DATA)
    matching_rows = jnp.where(mask)[0]#NIE SKOMPILUJE SIE ARGMAX
    ack_flag = jnp.where(matching_rows.size == 0, -1, 1)

    args = (action, buffer_state, ret_c, channel_state, no_tx, ack_flag)

    reward, ret_c, no_tx, is_ack = jax.lax.cond(action == Actions.TX.value, transmission, no_transmission, args)
    reward = jnp.where(terminal, 0., reward)

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
    obs, dest_address = jax.lax.cond(is_ack == 1, ack_sended, no_ack_sended, (obs, dest_address, matching_rows))

    obs_t = jnp.array([new_buffer_state, channel_state, ret_c, no_tx, power, dest_address, my_id, Ack_state.NOT_SENT])
    obs = jnp.roll(obs, -1, axis=0)
    obs = obs.at[-1].set(obs_t)

    return obs, reward, power


def process_output(buffer_states, new_buffer_states, power_states, channel_state, obs, actions, terminals):
    channel_states = jnp.full(buffer_states.shape[0], channel_state)
    return jax.vmap(process_output_i)(buffer_states, new_buffer_states, power_states, channel_states, obs, actions, terminals, dest_address, my_id)
