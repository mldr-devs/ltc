import unittest

import jax.numpy as jnp

from ltc.sim.constants import Actions, EMPTY_PACKET_ID
from ltc.sim.sim import *


class SimulateTestCase(unittest.TestCase):
    QUEUE = 4

    def test_channel_state_selector(self):
        # Test no actions
        actions = jnp.array([Actions.CS.value, Actions.CS.value, Actions.CS.value, Actions.CS.value])
        self.assertEqual(channel_state_selector(actions), 0)

        # Test single action
        actions = jnp.array([Actions.TX.value, Actions.CS.value, Actions.CS.value, Actions.CS.value])
        self.assertEqual(channel_state_selector(actions), 1)

        # Test multiple actions
        actions = jnp.array([Actions.TX.value, Actions.TX.value, Actions.CS.value, Actions.CS.value])
        self.assertEqual(channel_state_selector(actions), -1)

    def test_remove_transmitted_packets(self):
        buffer_states = jnp.array([
            [10, 11, 12, EMPTY_PACKET_ID],
            [20, 21, EMPTY_PACKET_ID, EMPTY_PACKET_ID],
        ])
        tx_masks = jnp.array([
            [1, 0, 1, 0],  # remove 1st and 3rd
            [0, 1, 0, 0],  # remove 2nd
        ])
        expected = jnp.array([
            [11, EMPTY_PACKET_ID, EMPTY_PACKET_ID, EMPTY_PACKET_ID],
            [20, EMPTY_PACKET_ID, EMPTY_PACKET_ID, EMPTY_PACKET_ID],
        ])
        result = remove_transmitted_packets(buffer_states, tx_masks)
        self.assertTrue(jnp.array_equal(result, expected))

    def test_add_new_frames(self):
        buffer_states = jnp.array([
            [EMPTY_PACKET_ID, EMPTY_PACKET_ID, EMPTY_PACKET_ID, EMPTY_PACKET_ID],
            [100, 101, EMPTY_PACKET_ID, EMPTY_PACKET_ID],
        ], dtype=jnp.int32)
        buffer_birth_steps = jnp.array([
            [-1, -1, -1, -1],
            [1, 2, -1, -1],
        ], dtype=jnp.int32)
        packet_seqs = jnp.array([0, 2], dtype=jnp.int32)
        new_frames = jnp.array([3, 5], dtype=jnp.int32)

        result, result_births, new_packet_seqs = add_new_frames(
            buffer_states, buffer_birth_steps, new_frames, packet_seqs, current_step=10
        )
        self.assertEqual(new_packet_seqs[0], 3)
        self.assertEqual(new_packet_seqs[1], 7)
        self.assertEqual(jnp.sum(result[0] != EMPTY_PACKET_ID), 3)
        # Overflow should cap to queue capacity and drop oldest first.
        self.assertEqual(jnp.sum(result[1] != EMPTY_PACKET_ID), self.QUEUE)
        self.assertEqual(int(result[1][0]), 13)
        self.assertEqual(int(result_births[1][0]), 10)

    def test_simulate(self):
        buffer_states = jnp.array([
            [7, 8, EMPTY_PACKET_ID, EMPTY_PACKET_ID],
            [EMPTY_PACKET_ID, EMPTY_PACKET_ID, EMPTY_PACKET_ID, EMPTY_PACKET_ID],
            [30, EMPTY_PACKET_ID, EMPTY_PACKET_ID, EMPTY_PACKET_ID],
            [40, 41, 42, EMPTY_PACKET_ID],
        ], dtype=jnp.int32)
        buffer_birth_steps = jnp.array([
            [1, 2, -1, -1],
            [-1, -1, -1, -1],
            [5, -1, -1, -1],
            [2, 3, 4, -1],
        ], dtype=jnp.int32)
        packet_seqs = jnp.array([2, 0, 1, 3], dtype=jnp.int32)
        new_frames = jnp.array([0, 2, 1, 0], dtype=jnp.int32)
        actions = jnp.array([Actions.TX.value, Actions.CS.value, Actions.CS.value, Actions.CS.value])

        (
            result_buffer_states,
            result_buffer_birth_steps,
            result_channel_state,
            result_packet_seqs,
            planned_packets,
            successful_packets,
        ) = simulate(
            buffer_states, buffer_birth_steps, new_frames, actions, packet_seqs, current_step=10
        )

        self.assertEqual(result_channel_state, 1)
        # Successful TX on station 0 removes queue head (7).
        self.assertEqual(int(result_buffer_states[0][0]), 8)
        self.assertTrue(jnp.array_equal(result_buffer_states[0][1:], jnp.array([EMPTY_PACKET_ID] * 3)))
        self.assertTrue(jnp.array_equal(result_buffer_birth_steps[0][1:], jnp.array([-1, -1, -1], dtype=jnp.int32)))
        # New arrivals were appended.
        self.assertEqual(jnp.sum(result_buffer_states[1] != EMPTY_PACKET_ID), 2)
        self.assertEqual(jnp.sum(result_buffer_states[2] != EMPTY_PACKET_ID), 2)
        self.assertTrue(jnp.array_equal(result_packet_seqs, packet_seqs + new_frames))
        self.assertTrue(jnp.array_equal(planned_packets, jnp.array([1, 0, 0, 0], dtype=jnp.int32)))
        self.assertTrue(jnp.array_equal(successful_packets, jnp.array([1, 0, 0, 0], dtype=jnp.int32)))


if __name__ == "__main__":
    unittest.main()
