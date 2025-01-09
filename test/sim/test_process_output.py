import unittest

import jax.numpy as jnp

from ltc.sim.constants import *
from ltc.sim.process_output import *


class ProcessOutputTestCase(unittest.TestCase):
    def test_no_transmission(self):
        args = (1, 5, 0)
        reward, buffer_state, r = no_transmission(args)
        self.assertEqual(reward, 0.0)
        self.assertEqual(buffer_state, 1)
        self.assertEqual(r, 5)

    def test_transmission_without_collision_empty_buffer(self):
        args = (0, 3, 0)
        reward, buffer_state, r = transmission_without_collision(args)
        self.assertEqual(reward, 0.0)
        self.assertEqual(buffer_state, 0)
        self.assertEqual(r, 3)

    def test_transmission_without_collision_successful(self):
        args = (1, 2, 0)
        reward, buffer_state, r = transmission_without_collision(args)
        self.assertAlmostEqual(reward, TX_REWARD / (2 + 1) ** 2)
        self.assertEqual(buffer_state, 0)
        self.assertEqual(r, 0)

    def test_transmission_with_collision_retransmission(self):
        args = (1, 2, 1)
        reward, buffer_state, r = transmission_with_collision(args)
        self.assertEqual(reward, COLLISION_PENALTY)
        self.assertEqual(buffer_state, 1)
        self.assertEqual(r, 3)

    def test_transmission_with_collision_max_retransmission(self):
        args = (1, MAX_RETRANSMISSION, 1)
        reward, buffer_state, r = transmission_with_collision(args)
        self.assertEqual(reward, COLLISION_PENALTY)
        self.assertEqual(buffer_state, 0)
        self.assertEqual(r, 0)

    def test_transmission_collision_path(self):
        args = (1, 7, 1)
        reward, buffer_state, r = transmission(args)
        self.assertEqual(reward, COLLISION_PENALTY)
        self.assertEqual(buffer_state, 1)
        self.assertEqual(r, 8)

    def test_transmission_successful_path(self):
        args = (1, 1, 0)
        reward, buffer_state, r = transmission(args)
        self.assertAlmostEqual(reward, TX_REWARD / (1 + 1) ** 2)
        self.assertEqual(buffer_state, 0)
        self.assertEqual(r, 0)

    def test_transmission_empty_buffer(self):
        args = (0, 1, 0)
        reward, buffer_state, r = transmission(args)
        self.assertEqual(reward, 0.0)
        self.assertEqual(buffer_state, 0)
        self.assertEqual(r, 1)

    def test_process_rl_output(self):
        buffer_states = jnp.array([0, 0, 1, 0])
        actions = jnp.array([1, 0, 0, 1])
        channel_state = 1
        obs_i_t_minus = jnp.array([
            [0, 0, 6],
            [0, 0, 7],
            [0, 0, 8]
        ])
        i = 0

        expected_buffer_states = jnp.array([0, 0, 1, 0])
        expected_obs_i_t = jnp.array([
            [0, 0, 7],
            [0, 0, 8],
            [0, 1, 0]
        ])
        expected_R_i = -1.0

        result_buffer_states, result_obs_i_t, result_R_i = process_rl_output(
            buffer_states, actions, channel_state, obs_i_t_minus, i
        )

        self.assertTrue(jnp.array_equal(result_buffer_states, expected_buffer_states))
        self.assertTrue(jnp.array_equal(result_obs_i_t, expected_obs_i_t))
        self.assertEqual(result_R_i, expected_R_i)


if __name__ == '__main__':
    unittest.main()
