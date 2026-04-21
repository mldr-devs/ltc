import jax
import jax.numpy as jnp

from ltc.sim.constants import Actions, EMPTY_PACKET_ID


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


def _cantor_pairing(a, b):
    s = a + b
    return (s * (s + 1)) // 2 + b


def is_buffer_empty(buffer_state):
    return jnp.all(buffer_state == EMPTY_PACKET_ID)


def remove_packets_from_queue(queue_state, remove_mask):
    """
    Removes packets selected by remove_mask and compacts remaining packets to queue head.
    """
    n = queue_state.shape[0]
    valid = queue_state != EMPTY_PACKET_ID
    remove_mask = jnp.asarray(remove_mask, dtype=bool)
    keep = valid & ~remove_mask

    # Stable compaction using argsort over static-size keys.
    idx = jnp.arange(n)
    sort_key = jnp.where(keep, idx, n + idx)
    order = jnp.argsort(sort_key)
    compacted = queue_state[order]
    kept_count = jnp.sum(keep.astype(jnp.int32))
    tail_mask = jnp.arange(n) >= kept_count
    return jnp.where(tail_mask, EMPTY_PACKET_ID, compacted)


def remove_transmitted_packets(buffer_states, tx_masks):
    return jax.vmap(remove_packets_from_queue)(buffer_states, tx_masks)


def enqueue_generated_packets(buffer_state, new_frames, station_id, packet_seq):
    """
    Appends generated packets at queue tail up to free capacity.
    """
    q = buffer_state.shape[0]
    new_frames = jnp.maximum(new_frames.astype(jnp.int32), 0)
    occupied = jnp.sum((buffer_state != EMPTY_PACKET_ID).astype(jnp.int32))
    capacity = q - occupied
    to_add = jnp.minimum(new_frames.astype(jnp.int32), capacity)

    idx = jnp.arange(q, dtype=jnp.int32)
    rel = idx - occupied
    add_mask = (idx >= occupied) & (rel < to_add)

    local_ids = packet_seq + jnp.maximum(rel.astype(jnp.int32), 0)
    packet_ids = _cantor_pairing(station_id.astype(jnp.int32), local_ids)
    next_state = jnp.where(add_mask, packet_ids, buffer_state)
    next_packet_seq = packet_seq + new_frames

    return next_state, next_packet_seq


def add_new_frames(buffer_states, new_frames, packet_seqs):
    station_ids = jnp.arange(buffer_states.shape[0], dtype=jnp.int32)
    return jax.vmap(enqueue_generated_packets)(buffer_states, new_frames, station_ids, packet_seqs)


def simulate(buffer_states, new_frames, actions, packet_seqs):
    """
    One simulation step with fixed-size packet queues.
    On successful TX, one packet from queue head is removed for transmitting station.
    """
    channel_state = channel_state_selector(actions)
    success_tx = (actions == Actions.TX.value) & (channel_state == 1)
    queue_size = buffer_states.shape[1]
    head_mask = jnp.arange(queue_size) == 0
    tx_masks = success_tx[:, None] & head_mask[None, :]

    buffer_states = remove_transmitted_packets(buffer_states, tx_masks)
    buffer_states, packet_seqs = add_new_frames(buffer_states, new_frames, packet_seqs)
    return buffer_states, channel_state, packet_seqs
