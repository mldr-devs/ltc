import unittest

import jax.numpy as jnp

from ltc.sim.constants import Actions
from ltc.sim.sim import *


class SimulateTestCase(unittest.TestCase):
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

    def test_buffer_clearing(self):
        # Test clearing buffers when actions match
        buffer_states = jnp.array([1, 0, 1, 0])
        actions = jnp.array([Actions.TX.value, Actions.CS.value, Actions.CS.value, Actions.CS.value])
        expected = jnp.array([0, 0, 1, 0])
        result = buffer_clearing(buffer_states, actions)
        self.assertTrue(jnp.array_equal(result, expected))

        # Test no clearing when no matching actions
        buffer_states = jnp.array([1, 0, 1, 0])
        actions = jnp.array([Actions.CS.value, Actions.CS.value, Actions.CS.value, Actions.CS.value])
        expected = jnp.array([1, 0, 1, 0])
        result = buffer_clearing(buffer_states, actions)
        self.assertTrue(jnp.array_equal(result, expected))

    def test_add_new_frames(self):
        # Test adding new frames
        buffer_states = jnp.array([1, 0, 1, 0])
        new_frames = jnp.array([0, 1, 0, 0])
        expected = jnp.array([1, 1, 1, 0])
        result = add_new_frames(buffer_states, new_frames)
        self.assertTrue(jnp.array_equal(result, expected))

        # Test no new frames added
        buffer_states = jnp.array([1, 0, 1, 0])
        new_frames = jnp.array([0, 0, 0, 0])
        expected = jnp.array([1, 0, 1, 0])
        result = add_new_frames(buffer_states, new_frames)
        self.assertTrue(jnp.array_equal(result, expected))

    def test_simulate(self):
        # Test full simulation
        buffer_states = jnp.array([1, 1, 1, 0])
        new_frames = jnp.array([0, 1, 1, 1])
        actions = jnp.array([Actions.TX.value, Actions.CS.value, Actions.CS.value, Actions.CS.value])

        expected_buffer_states = jnp.array([0, 1, 1, 1])
        expected_channel_state = 1

        result_buffer_states, result_channel_state = simulate(buffer_states, new_frames, actions)
        self.assertTrue(jnp.array_equal(result_buffer_states, expected_buffer_states))
        self.assertEqual(result_channel_state, expected_channel_state)

        # Test simulation with no actions
        buffer_states = jnp.array([1, 0, 1, 0])
        new_frames = jnp.array([0, 1, 0, 1])
        actions = jnp.array([Actions.CS.value, Actions.CS.value, Actions.CS.value, Actions.CS.value])

        expected_buffer_states = jnp.array([1, 1, 1, 1])
        expected_channel_state = 0

        result_buffer_states, result_channel_state = simulate(buffer_states, new_frames, actions)
        self.assertTrue(jnp.array_equal(result_buffer_states, expected_buffer_states))
        self.assertEqual(result_channel_state, expected_channel_state)

        # Test simulation with multiple actions
        buffer_states = jnp.array([0, 0, 0, 0])
        new_frames = jnp.array([0, 1, 0, 1])
        actions = jnp.array([Actions.TX.value, Actions.CS.value, Actions.CS.value, Actions.TX.value])

        expected_buffer_states = jnp.array([0, 1, 0, 1])
        expected_channel_state = -1

        result_buffer_states, result_channel_state = simulate(buffer_states, new_frames, actions)
        self.assertTrue(jnp.array_equal(result_buffer_states, expected_buffer_states))
        self.assertEqual(result_channel_state, expected_channel_state)


if __name__ == "__main__":
    unittest.main()
