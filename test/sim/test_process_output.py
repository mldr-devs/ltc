import unittest

import jax.numpy as jnp

from ltc.sim.constants import *
from ltc.sim.process_output import *


class ProcessOutputTestCase(unittest.TestCase):
    def test_no_transmission(self):
        args = (1, 1, 5, 1, 0, 0)
        reward, r, no_tx = no_transmission(args)
        self.assertEqual(reward, 0.0)
        self.assertEqual(r, 5)

    def test_transmission_without_collision_empty_buffer(self):
        args = (1, 0, 0, 1, 0, 0)
        reward, r, no_tx = transmission_without_collision(args)
        self.assertEqual(reward, EMPTY_TX_PENALTY)
        self.assertEqual(r, 0)

    def test_transmission_without_collision_successful(self):
        args = (1, 1, 2, 1, 0, 0)
        reward, r, no_tx = transmission_without_collision(args)
        self.assertAlmostEqual(reward, TX_REWARD / (2 + 1))
        self.assertEqual(r, 0)

    def test_transmission_with_collision_retransmission(self):
        args = (1, 1, 2, -1, 0, 0)
        reward, r, no_tx = transmission_with_collision(args)
        self.assertEqual(reward, COLLISION_PENALTY)
        self.assertEqual(r, 3)

    def test_transmission_with_collision_max_retransmission(self):
        args = (1, 1, MAX_RETRANSMISSION, -1, 0, 0)
        reward, r, no_tx = transmission_with_collision(args)
        self.assertEqual(reward, MAX_RETRANSMISSION_PENALTY)
        self.assertEqual(r, 0)

    def test_transmission_collision_path(self):
        args = (1, 1, 7, -1, 0, 0)
        reward, r, no_tx = transmission(args)
        self.assertEqual(reward, COLLISION_PENALTY)
        self.assertEqual(r, 8)

    def test_transmission_successful_path(self):
        args = (1, 1, 1, 1, 0, 0)
        reward, r, no_tx = transmission(args)
        self.assertAlmostEqual(reward, TX_REWARD / (1 + 1))
        self.assertEqual(r, 0)

    def test_transmission_empty_buffer(self):
        args = (1, 0, 1, 1, 0, 0)
        reward, r, no_tx = transmission(args)
        self.assertEqual(reward, EMPTY_TX_PENALTY)
        self.assertEqual(r, 0)

    def test_process_rl_output(self):
        buffer_states = jnp.array([1, 1, 1, 1])
        new_buffer_states = jnp.array([0, 1, 1, 0])
        power_states = jnp.array([100, 100, 100, 100])
        actions = jnp.array([Actions.ACK.value, Actions.IDLE.value, Actions.TX.value, Actions.TX.value])
        terminals = jnp.array([False, False, False, False])
        n = 4
        window_size = 4
        channel_state = -1
        obs = jnp.array([[
            [1, 1, 6, 0, 100, 0],
            [1, 1, 7, 0, 100, 0],
            [1, 1, 8, 0, 100, 0]
        ]] * n)

        id_values = jnp.arange(0, n).reshape(n, 1)
        obs = obs.at[:, :, Observation_indexes.ID_INDEX.value].set(id_values)
        expected_obs_0 = jnp.array([
            [1, 1, 7, 0, 100, 0],
            [1, 1, 8, 0, 100, 0],
            [0, -1, 0, 0, 100 - ACK_CONSUMPTION, 0],
        ])
        expected_R_0 = -1.0
        expected_power = power_states + jnp.array([-ACK_CONSUMPTION, -IDLE_CONSUMPTION, -TX_CONSUMPTION, -TX_CONSUMPTION])

        transmission_history = jnp.array([[0, 1],
                                          [0, 0],
                                          [0, 1],
                                          [3, 0],
                                          [0, 1]])

        result_obs, result_R, power = process_output(
            buffer_states, new_buffer_states, power_states, channel_state, obs, actions, terminals, transmission_history
        )


        self.assertTrue(jnp.array_equal(result_obs[0], expected_obs_0))
        self.assertEqual(result_R[0], expected_R_0)
        self.assertTrue(jnp.array_equal(power, expected_power))


if __name__ == '__main__':
    unittest.main()
