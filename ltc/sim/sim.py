import jax
import jax.numpy as jnp

from ltc.sim.constants import Actions, EMPTY_PACKET_ID


def channel_state_selector(actions):
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
    n = queue_state.shape[0]
    valid = queue_state != EMPTY_PACKET_ID
    remove_mask = jnp.asarray(remove_mask, dtype=bool)
    keep = valid & ~remove_mask

    idx = jnp.arange(n)
    sort_key = jnp.where(keep, idx, n + idx)
    order = jnp.argsort(sort_key)
    compacted = queue_state[order]
    kept_count = jnp.sum(keep.astype(jnp.int32))
    tail_mask = jnp.arange(n) >= kept_count
    return jnp.where(tail_mask, EMPTY_PACKET_ID, compacted)


def remove_transmitted_packets(buffer_states, tx_masks):
    return jax.vmap(remove_packets_from_queue)(buffer_states, tx_masks)


def enqueue_generated_packets(buffer_state, buffer_birth_state, new_frames, station_id, packet_seq, current_step):
    q = buffer_state.shape[0]
    new_frames = jnp.maximum(new_frames.astype(jnp.int32), 0)
    occupied = jnp.sum((buffer_state != EMPTY_PACKET_ID).astype(jnp.int32))

    kept_new = jnp.minimum(new_frames, q)
    kept_old = jnp.minimum(occupied, q - kept_new)
    old_start = occupied - kept_old
    new_start = packet_seq + (new_frames - kept_new)

    idx = jnp.arange(q, dtype=jnp.int32)

    take_old = idx < kept_old
    old_idx = jnp.clip(old_start + idx, 0, jnp.maximum(occupied - 1, 0))
    old_packets = jnp.where(take_old, buffer_state[old_idx], EMPTY_PACKET_ID)
    old_births = jnp.where(take_old, buffer_birth_state[old_idx], -1)

    new_rel = idx - kept_old
    take_new = jnp.logical_and(idx >= kept_old, new_rel < kept_new)
    new_local_ids = new_start + jnp.maximum(new_rel, 0)
    new_packets = _cantor_pairing(station_id.astype(jnp.int32), new_local_ids)
    new_births = jnp.full((q,), current_step, dtype=jnp.int32)

    next_state = jnp.where(take_old, old_packets, jnp.where(take_new, new_packets, EMPTY_PACKET_ID))
    next_birth = jnp.where(take_old, old_births, jnp.where(take_new, new_births, -1))
    next_packet_seq = packet_seq + new_frames
    return next_state, next_birth, next_packet_seq


def add_new_frames(buffer_states, buffer_birth_steps, new_frames, packet_seqs, current_step):
    station_ids = jnp.arange(buffer_states.shape[0], dtype=jnp.int32)
    return jax.vmap(enqueue_generated_packets, in_axes=(0, 0, 0, 0, 0, None))(
        buffer_states, buffer_birth_steps, new_frames, station_ids, packet_seqs, current_step
    )


def simulate(buffer_states, buffer_birth_steps, new_frames, actions, packet_seqs, current_step):
    channel_state = channel_state_selector(actions)
    pre_nonempty = jnp.any(buffer_states != EMPTY_PACKET_ID, axis=1)
    success_tx = (actions == Actions.TX.value) & (channel_state == 1)
    planned_packets = (actions == Actions.TX.value).astype(jnp.int32)
    successful_packets = (success_tx & pre_nonempty).astype(jnp.int32)
    queue_size = buffer_states.shape[1]
    head_mask = jnp.arange(queue_size) == 0
    tx_masks = success_tx[:, None] & head_mask[None, :]

    buffer_states = remove_transmitted_packets(buffer_states, tx_masks)
    buffer_birth_steps = remove_transmitted_packets(buffer_birth_steps, tx_masks)
    buffer_states, buffer_birth_steps, packet_seqs = add_new_frames(
        buffer_states, buffer_birth_steps, new_frames, packet_seqs, current_step
    )
    return (
        buffer_states,
        buffer_birth_steps,
        channel_state,
        packet_seqs,
        planned_packets,
        successful_packets,
    )
