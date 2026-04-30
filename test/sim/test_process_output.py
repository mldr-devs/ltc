import unittest

import jax
import jax.numpy as jnp

from ltc.sim.constants import *
from ltc.sim.process_output import *
from ltc.utils.structs import ObsFeatureInputs, ObsFeatureConfig

KEY = jax.random.PRNGKey(0)
EMPTY = EMPTY_PACKET_ID


class ProcessOutputTestCase(unittest.TestCase):
    def test_no_transmission(self):
        args = (Actions.IDLE.value, jnp.array([1, EMPTY, EMPTY]), 5, 0, 1, KEY)
        reward, r, no_tx = no_transmission(args)
        self.assertEqual(reward, 0.0)
        self.assertEqual(r, 5)

    def test_transmission_with_collision_retransmission(self):
        args = (Actions.TX.value, jnp.array([5, EMPTY, EMPTY]), 2, -1, 1, KEY)
        reward, r, no_tx = transmission_with_collision(args)
        self.assertEqual(reward, 0.0)
        self.assertEqual(r, 3)

    def test_transmission_with_collision_max_retransmission(self):
        args = (Actions.TX.value, jnp.array([5, EMPTY, EMPTY]), MAX_RETRANSMISSION, -1, 1, KEY)
        reward, r, no_tx = transmission_with_collision(args)
        self.assertEqual(reward, 0.0)
        self.assertEqual(r, 0)

    def test_transmission_collision_path(self):
        args = (Actions.TX.value, jnp.array([5, EMPTY, EMPTY]), 7, -1, 1, KEY)
        reward, r, no_tx = transmission(args)
        self.assertEqual(reward, 0.0)
        self.assertEqual(r, 8)

    def test_transmission_successful_path(self):
        args = (Actions.TX.value, jnp.array([5, EMPTY, EMPTY]), 1, 1, 1, KEY)
        reward, r, no_tx = transmission(args)
        self.assertEqual(reward, 0.0)
        self.assertEqual(r, 0)

    def test_transmission_empty_buffer(self):
        args = (Actions.TX.value, jnp.array([EMPTY, EMPTY, EMPTY]), 1, 1, 1, KEY)
        reward, r, no_tx = transmission(args)
        self.assertEqual(reward, 0.0)
        self.assertEqual(r, 0)

    def test_process_rl_output(self):
        buffer_states = jnp.array([
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [9, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ])
        new_buffer_birth_states = jnp.array([
            [-1, -1, -1],
            [-1, -1, -1],
            [3, -1, -1],
            [-1, -1, -1],
        ], dtype=jnp.int32)
        new_buffer_states = jnp.array([
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [9, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ])
        power_states = jnp.array([100, 100, 100, 100])
        actions = jnp.array([Actions.TX.value, Actions.CS.value, Actions.TX.value, Actions.TX.value])
        terminals = jnp.array([False, False, False, False])
        channel_state = -1
        obs = jnp.zeros((4, 3, OBS_SIZE), dtype=jnp.float32)
        obs = obs.at[:, -1, ObsIdx.STATUS_RETRY_COUNTER].set(8)
        expected_R_0 = -1.0
        expected_power = power_states + jnp.array([-TX_CONSUMPTION, -CS_CONSUMPTION, -TX_CONSUMPTION, -TX_CONSUMPTION])

        obs_features = ObsFeatureInputs(
            traffic_mean_arrival_rate=jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
            channel_occupancy_pct_window=jnp.full((4,), 0.25, dtype=jnp.float32),
            channel_collisions_pct_window=jnp.full((4,), 0.5, dtype=jnp.float32),
            back_pct=jnp.array([0.0, 0.6, 1.0, 0.0], dtype=jnp.float32),
            unique_ltc_tx_window=jnp.full((4,), 2.0, dtype=jnp.float32),
            cs_tx_same_type_now=jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float32),
            tx_collision_other_now=jnp.array([1.0, 0.0, 1.0, 1.0], dtype=jnp.float32),
        )
        obs_config = ObsFeatureConfig(
            enable_cs_tx_same_type=True,
            enable_tx_collision_other=True,
        )

        # Agent 0: TX, collision, ret_c=8 >= MAX_RETRANSMISSION, coex collision → TX_MAX_RETRANSMISSION_PENALTY
        done_now = jnp.array([True, True, True, True])
        k_tx = jnp.zeros(4, dtype=jnp.float32)
        k_coll_ltc = jnp.zeros(4, dtype=jnp.float32)
        k_coll_coex = jnp.array([1.0, 0.0, 1.0, 1.0], dtype=jnp.float32)  # coex collision for TX agents
        header_collision = jnp.zeros(4, dtype=bool)
        header_collision_other = jnp.zeros(4, dtype=bool)
        ack_collision = jnp.zeros(4, dtype=bool)
        ack_collision_other = jnp.zeros(4, dtype=bool)
        initial_ret_c = jnp.array([8, 0, 8, 8], dtype=jnp.int32)
        tx_started_empty = jnp.array([True, False, False, True], dtype=bool)

        result_obs, result_R, power = process_output(
            buffer_states=buffer_states,
            new_buffer_states=new_buffer_states,
            new_buffer_birth_states=new_buffer_birth_states,
            power_states=power_states,
            channel_state=channel_state,
            obs=obs,
            actions=actions,
            terminals=terminals,
            key=KEY,
            current_step=10,
            obs_features=obs_features,
            obs_config=obs_config,
            done_now=done_now,
            k_tx=k_tx,
            k_coll_ltc=k_coll_ltc,
            k_coll_coex=k_coll_coex,
            header_collision=header_collision,
            header_collision_other=header_collision_other,
            ack_collision=ack_collision,
            ack_collision_other=ack_collision_other,
            initial_ret_c=initial_ret_c,
            tx_started_empty=tx_started_empty,
        )

        self.assertEqual(result_obs.shape[-1], OBS_SIZE)
        self.assertEqual(result_obs[1, -1, ObsIdx.CHANNEL_LAST_CS_BUSY], 1.0)
        self.assertEqual(result_obs[1, -1, ObsIdx.CHANNEL_LAST_CS_TX_SAME_TYPE], -1.0)
        self.assertEqual(result_obs[0, -1, ObsIdx.CHANNEL_LAST_TX_COLLISION_OTHER], 1.0)
        self.assertEqual(result_obs[2, -1, ObsIdx.BUFFER_PACKET_COUNT], 1.0)
        self.assertEqual(result_obs[2, -1, ObsIdx.ACTION_BACK_PCT], 1.0)
        self.assertEqual(result_R[0], expected_R_0)
        self.assertTrue(jnp.array_equal(power, expected_power))

if __name__ == '__main__':
    unittest.main()
