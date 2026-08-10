import itertools
import unittest

import jax
import jax.numpy as jnp

from ltc.sim.constants import Actions, Features
from ltc.sim.sim import *

KEY = jax.random.PRNGKey(0)


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

        result_buffer_states, result_channel_state, _ = simulate(
            buffer_states, new_frames, actions, KEY, error_probability=0.0
        )
        self.assertTrue(jnp.array_equal(result_buffer_states, expected_buffer_states))
        self.assertEqual(result_channel_state, expected_channel_state)

        # Test simulation with no actions
        buffer_states = jnp.array([1, 0, 1, 0])
        new_frames = jnp.array([0, 1, 0, 1])
        actions = jnp.array([Actions.CS.value, Actions.CS.value, Actions.CS.value, Actions.CS.value])

        expected_buffer_states = jnp.array([1, 1, 1, 1])
        expected_channel_state = 0

        result_buffer_states, result_channel_state, _ = simulate(
            buffer_states, new_frames, actions, KEY, error_probability=0.0
        )
        self.assertTrue(jnp.array_equal(result_buffer_states, expected_buffer_states))
        self.assertEqual(result_channel_state, expected_channel_state)

        # Test simulation with multiple actions
        buffer_states = jnp.array([0, 0, 0, 0])
        new_frames = jnp.array([0, 1, 0, 1])
        actions = jnp.array([Actions.TX.value, Actions.CS.value, Actions.CS.value, Actions.TX.value])

        expected_buffer_states = jnp.array([0, 1, 0, 1])
        expected_channel_state = -1

        result_buffer_states, result_channel_state, _ = simulate(
            buffer_states, new_frames, actions, KEY, error_probability=0.0
        )
        self.assertTrue(jnp.array_equal(result_buffer_states, expected_buffer_states))
        self.assertEqual(result_channel_state, expected_channel_state)


class ActionConstraintTestCase(unittest.TestCase):
    def test_apply_empty_buffer_constraint(self):
        actions = jnp.array([Actions.TX.value, Actions.TX.value, Actions.CS.value, Actions.IDLE.value])
        buffer_states = jnp.array([1, 0, 0, 0])
        expected = jnp.array([Actions.TX.value, Actions.CS.value, Actions.CS.value, Actions.IDLE.value])
        self.assertTrue(jnp.array_equal(apply_empty_buffer_constraint(actions, buffer_states), expected))

    def test_apply_lbt_constraint(self):
        # Only the station that sensed an idle channel last slot may transmit.
        prev_obs = jnp.zeros((3, 1, len(Features)), dtype=int)
        prev_obs = prev_obs.at[:, -1, Features.CHANNEL].set(jnp.array([0, 1, 0]))
        prev_actions = jnp.array([Actions.CS.value, Actions.CS.value, Actions.TX.value])
        actions = jnp.array([Actions.TX.value, Actions.TX.value, Actions.TX.value])

        expected = jnp.array([Actions.TX.value, Actions.CS.value, Actions.CS.value])
        self.assertTrue(jnp.array_equal(apply_lbt_constraint(actions, prev_actions, prev_obs), expected))


class SuccessfulStationTestCase(unittest.TestCase):
    def test_reports_the_transmitter_on_success(self):
        actions = jnp.array([Actions.CS.value, Actions.TX.value, Actions.IDLE.value])
        self.assertEqual(get_successful_station_id(actions, 1), 1)

    def test_reports_none_on_collision_or_idle(self):
        actions = jnp.array([Actions.TX.value, Actions.TX.value, Actions.IDLE.value])
        self.assertEqual(get_successful_station_id(actions, -1), -1)
        self.assertEqual(get_successful_station_id(jnp.zeros(3, dtype=int) + Actions.CS.value, 0), -1)


class HiddenStationTestCase(unittest.TestCase):
    def test_full_visibility_matches_the_true_channel_state(self):
        n = 4
        ones = jnp.ones((n, n), dtype=int)
        for actions in itertools.product([a.value for a in Actions], repeat=n):
            actions = jnp.array(actions)
            for phy_error in (0, 1):
                channel_state = jnp.where(phy_error == 1, -1, channel_state_selector(actions))
                observed = hidden_station_observation(ones, channel_state, actions)
                self.assertTrue(jnp.all(observed == channel_state))

    def test_hidden_transmitter_looks_idle(self):
        # Stations 0 and 1 cannot hear each other, so each perceives its own slot as a success.
        visibility = jnp.array([[1, 0], [0, 1]])
        actions = jnp.array([Actions.TX.value, Actions.TX.value])
        observed = hidden_station_observation(visibility, channel_state_selector(actions), actions)
        self.assertTrue(jnp.array_equal(observed, jnp.array([-1, -1])))

    def test_silent_station_hears_nothing_from_a_hidden_peer(self):
        # Station 2 hears nobody; stations 0 and 1 hear each other.
        visibility = jnp.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
        actions = jnp.array([Actions.TX.value, Actions.CS.value, Actions.CS.value])
        observed = hidden_station_observation(visibility, channel_state_selector(actions), actions)
        self.assertEqual(observed[0], 1)
        self.assertEqual(observed[1], 1)
        self.assertEqual(observed[2], 0)


if __name__ == "__main__":
    unittest.main()
